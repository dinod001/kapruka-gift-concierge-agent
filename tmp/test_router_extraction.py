import os
import sys
from dotenv import load_dotenv

PROJECT_ROOT = r"E:\my career life\AI Enginert Essential\All Mini Projects\Mini project 03"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from infrastructure.llm.llm_provider import get_router_llm
from agents.router import QueryRouter

router = QueryRouter(llm=get_router_llm())

msg = "i need another difference phone can you suggest"
ctx = "Assistant: I recommend the Nubia Neo 3 5G 8GB 256GB Shadow Black. It is a great 5G phone for RS.70,000."

print(f"Testing Routing for: '{msg}'")
decision = router.route(msg, memory_context=ctx)
print(f"Decision: {decision.route}")
print(f"Params: {decision.params}")
