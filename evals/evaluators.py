from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from src.config import settings
import structlog

log = structlog.get_logger()

class JudgeScore(BaseModel):
    relevance: int = Field(ge=1, le=5, description="1-5 how relevant the answer is to the question")
    grounded: int = Field(ge=1, le=5, description="1-5 how well claims are supported by citations")
    clarity: int = Field(ge=1, le=5)
    reasoning: str = Field(description="2-3 sentences explaining the scores in plain English")

judge_llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_chat_model,
    temperature=0.0,
).with_structured_output(JudgeScore)

JUDGE_PROMPT = """You are an impartial evaluator of a research assistant.

Question: {q}
Answer: {a}
Papers cited: {p}

Score 1-5:
- relevance: does the answer address the question?
- grounded: are claims attributable to the papers (citations present, consistent)?
- clarity: is it well-written and organized?"""

def judge(question: str, answer: str, papers: list) -> JudgeScore:
    try:
        return judge_llm.invoke(JUDGE_PROMPT.format(q=question, a=answer, p=papers))
    except Exception as e:
        log.error("judge_failed", error=str(e), question=question[:80])
        return None