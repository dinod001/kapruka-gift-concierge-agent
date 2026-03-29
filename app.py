"""
Kapruka Gift-Concierge — Flask Web App
=======================================
Uses a Queue + Thread pattern so on_step callbacks from the orchestrator
are forwarded to the browser in real-time via Server-Sent Events (SSE).

Real-time pipeline events:
  memory        → 🧠 LT profiles loaded / not found
  route         → 🔀 Route decided (logistic / rag / direct)
  draft         → ⚙️  Draft generation started
  reflecting    → 🔍 Reflection started
  reflect_result→ ✅ Safe / ⚠️ Violation reason
  revised       → ✏️  Draft revised
  done          → Final answer + metadata
"""

import os
import sys
import json
import queue
import threading

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from flask import Flask, render_template, request, Response, jsonify, stream_with_context, redirect, url_for

from infrastructure.llm.llm_provider import (
    get_chat_llm,
    get_router_llm,
    get_extractor_llm,
)
from services.ingest_service.embeddings import get_default_embeddings
from memory.memory_ops import MemoryDistiller
from agents.tools.logsitic_agent import LogisticAlert
from agents.tools.rag_tool import RAGTool
from agents.orchestrator import AgentOrchestrator

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

# ── Initialise agent once at startup ──────────────────────────────────────────
llm_chat      = get_chat_llm(temperature=0.3)
llm_router    = get_router_llm(temperature=0)
llm_extractor = get_extractor_llm(temperature=0)

embedder      = get_default_embeddings()

profile_path  = os.path.join(PROJECT_ROOT, "data", "recipient_profiles.json")
memory        = MemoryDistiller(llm=llm_extractor, profile_path=profile_path)
logistic_tool = LogisticAlert(llm=llm_extractor)
rag_tool      = RAGTool(embedder=embedder, llm=llm_chat)

agent = AgentOrchestrator(
    llm_chat=llm_chat,
    llm_router=llm_router,
    memory=memory,
    logistic_tool=logistic_tool,
    rag_tool=rag_tool,
)

_SENTINEL = object()          # marks end of queue stream


def _sse(data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"data: {json.dumps(data)}\n\n"


def _build_on_step(q: queue.Queue):
    """
    Returns a callback that maps orchestrator step events → SSE payloads
    and puts them into the queue for the SSE generator to read.
    """
    def on_step(name: str, payload: dict) -> None:

        if name == "memory":
            count = payload.get("count", 0)
            found = payload.get("found", False)
            q.put(_sse({
                "step":  "memory",
                "found": found,
                "count": count,
                "msg":   f"✅ Memory Found ({count} profile{'s' if count != 1 else ''})"
                          if found else "○ No Long-Term Memory",
            }))

        elif name == "route":
            route = payload.get("route", "direct")
            conf  = payload.get("confidence", 0)
            q.put(_sse({
                "step":       "route",
                "route":      route,
                "confidence": conf,
                "msg":        f"🔀 Route → {route.upper()} ({int(conf * 100)}%)",
            }))

        elif name == "draft":
            q.put(_sse({"step": "draft", "msg": "⚙️  Generating draft…"}))

        elif name == "reflecting":
            count = payload.get("profile_count", 0)
            q.put(_sse({
                "step": "reflecting",
                "msg":  f"🔍 Reflecting against {count} profile(s)…",
            }))

        elif name == "reflect_result":
            violated = payload.get("violated", False)
            reason   = payload.get("reason", "")
            q.put(_sse({
                "step":     "reflect_result",
                "violated": violated,
                "reason":   reason,
                "msg":      f"⚠️  Violation: {reason}" if violated else "✅ Reflection Safe",
            }))

        elif name == "revised":
            q.put(_sse({
                "step": "revised",
                "msg":  "✏️  Draft revised for safety",
            }))

    return on_step


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return redirect(url_for("production"))


@app.get("/develop")
def develop():
    return render_template("develop.html")


@app.get("/production")
def production():
    return render_template("production.html")

@app.post("/chat")
def chat():
    body    = request.get_json(force=True)
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    q = queue.Queue()

    def run_agent():
        """Run orchestrator in a background thread, putting SSE frames in queue."""
        try:
            on_step  = _build_on_step(q)
            response = agent.chat(message, on_step=on_step)
            # Final "done" event carries the answer
            q.put(_sse({
                "step":     "done",
                "answer":   response.answer,
                "route":    response.route,
                "violated": response.violated,
                "latency":  response.latency_ms,
            }))
        except Exception as exc:
            q.put(_sse({"step": "error", "msg": str(exc)}))
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    def event_stream():
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            yield item

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
def api_chat():
    """
    Simple blocking JSON endpoint for the production UI.
    No SSE — waits for the full response then returns JSON.
    """
    body    = request.get_json(force=True)
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    try:
        response = agent.chat(message)
        return jsonify({
            "answer":   response.answer,
            "route":    response.route,
            "violated": response.violated,
            "latency":  response.latency_ms,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/profiles")
def get_profiles():
    data = memory.long_term.get_profiles()
    return jsonify({"profiles": data, "count": len(data)})


@app.delete("/profiles")
def clear_profiles():
    memory.long_term.save_profiles([])
    return jsonify({"status": "cleared"})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=True lets Flask handle each request in its own thread,
    # which is required for the SSE + background-thread pattern.
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
