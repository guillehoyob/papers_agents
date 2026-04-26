from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=2000)

class PaperRef(BaseModel):
    paper_id: str
    title: str
    source: str

class ChatResponse(BaseModel):
    answer: str
    papers_use: list[PaperRef]
    tool_calls: list[str]
    