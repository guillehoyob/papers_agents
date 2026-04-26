from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
import structlog, uuid
from src.graph import build_graph
from src.api.schemas import ChatRequest, ChatResponse, PaperRef

log = structlog.get_logger()
app = FastAPI(title="Research Assistant")
graph = build_graph(with_memory=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    trace_id = str(uuid.uuid4())[:8]
    log.info("chat_start", trace=trace_id, thread=req.thread_id)
    cfg = {"configurable": {"thread_id": req.thread_id}}
    try:
        state = graph.invoke(
            {"messages": [HumanMessage(content=req.message)],
            "papers_found": [], "research_complete": False,
            "tool_calls_made": []},
            config=cfg,
        )
    except Exception as e:
        log.error("chat_failed", trace=trace_id, error=str(e))
        raise HTTPException(500, f"Internal error: {e}")

    final = state["messages"][-1].content
    papers = [
        PaperRef(
            paper_id=p.get("paper_id", "?"),
            title=p.get("paper_title") or p.get("title", "?"),
            source="arxiv" if "abstract" in p else "local",
        )
        for p in state.get("papers_found", [])
    ]
    log.info("chat_done", trace=trace_id, papers=len(papers))
    return ChatResponse(
        answer=final,
        papers_use=papers,
        tool_calls=state.get("tool_calls_made", []),
    )