import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from src.config import settings

llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_chat_model,
    temperature=0.0
)

print(settings.ollama_base_url,settings.ollama_chat_model,)
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant. Answer briefly."),
    HumanMessage(content="What is RAG in one sentence?")
])
print("Basic:", response.content)

class PaperSummary(BaseModel):
    title: str = Field(description="Concise title")
    key_ideas: list[str] = Field(description="3 main ideas")
    confidence: float = Field(description="0-1 confidence")

structured_llm = llm.with_structured_output(PaperSummary)
result = structured_llm.invoke("Summarize: Retrieval Argumented Generation combines retrieval with generation")
print("Structured:", result)