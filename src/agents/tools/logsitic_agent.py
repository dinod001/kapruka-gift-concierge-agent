from loguru import logger

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agents.prompts.agent_prompts import LOGISTIC_PROMPT


class LogisticAlert:

    def __init__(self, llm, prompt: str = LOGISTIC_PROMPT) -> None:
        self.llm    = llm
        self.prompt = prompt

    def generate_answer(self, question: str) -> str:
        """
        Check delivery feasibility for a given Sri Lankan district.

        Parameters
        ----------
        question : str
            User question, e.g. "Can you deliver to Jaffna?"

        Returns
        -------
        str
            Formatted feasibility response from the LLM.
        """
        logger.info(f"[LogisticAlert] Checking delivery feasibility: '{question}'")

        prompt = PromptTemplate(
            input_variables=["question"],
            template=self.prompt,
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question})

        logger.success(f"[LogisticAlert] Response generated.")
        return answer
