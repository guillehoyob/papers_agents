from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from src.config import settings
from src.state import AgentState
from src.tools.search_local import search_local_papers
from src.tools.search_arxiv import search_arxiv_online
from src.agents.prompts import RESEARCH_SYSTEM

TOOLS = [search_local_papers, search_arxiv_online]

llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_chat_model,
    temperature=0.0,
)
llm_with_tools = llm.bind_tools(TOOLS)

def research_node(state: AgentState) -> dict:
    """LLM call with tools boud. Decides whether to call a tool or stop."""
    messages = [SystemMessage(content=RESEARCH_SYSTEM)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    # Extract any new papers from recent tool results to populate state
    papers = state.get("papers_found", [])
    tool_calls_log = state.get("tool_calls_made", [])
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls_log.append(tc["name"])
    
    # Did the agent signal compeltion?
    complete = (not response.tool_calls and "RESEARCH_DONE" in (response.content or ""))

    return {
        "messages": [response],
        "tool_calls_made": tool_calls_log,
        "research_compelte": complete,
    }

tool_node = ToolNode(TOOLS)

def extract_papers_from_tools(state: AgentState) -> dict:
    """After a tool call, harvest structured paper info into state."""
    from langchain_core.messages import ToolMessage
    papers = list(state.get("papers_found", []))
    known_ids = {p["paper_id"] for p in papers}
    for m in reversed(state["messages"]):
        if not isinstance(m, ToolMessage): break
        # m.content is the tool's return value (list of dicts, JSON-serialized)
        import json
        try:
            items = json.loads(m.content) if isinstance(m.content, str) else m.content
        except Exception:
            continue
        for it in items:
            pid = it.get("paper_id")
            if pid and pid not in known_ids:
                papers.append(it)
                known_ids.add(pid)
    return {"papers_found": papers}