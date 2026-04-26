from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from sympy.assumptions.assume import true
from src.state import AgentState
from src.agents.research import research_node, tool_node, extract_papers_from_tools
from src.agents.analysis import analysis_node
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from src.config import settings

def build_graph(with_memory: bool = True):
    g = StateGraph(AgentState)
    g.add_node("research", research_node)
    g.add_node("tools", tool_node)
    g.add_node("harvest", extract_papers_from_tools)
    g.add_node("analysis", analysis_node)

    g.add_edge(START, "research")

    def after_research(state):
        last = state["messages"][-1]
        # If research agent wants to call a tool -> tools
        if getattr(last, "tool_calls", None):
            return "tools"

        # Otherwise it's done -> pass to analysis
        return "analysis"

    # tools_condition: if last AI msg has tool_calls -> "tools", else -> END
    g.add_conditional_edges("research", after_research, {
        "tools": "tools",
        "analysis": "analysis"
    })
    g.add_edge("analysis", END)
    g.add_edge("tools", "harvest")
    g.add_edge("harvest", "research") # loop back
    
    if with_memory:
        conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        return g.compile(checkpointer=checkpointer)
    return g.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from src.graph import build_graph

    app = build_graph()
    cfg = {"configurable": {"thread_id": "user-42"}}
    result = app.invoke({
        "messages": [HumanMessage(content="What does LoRA propose for efficient fine-tunning?")],
        "papers_found": [], "research_complete": False, "tool_call_made": [],
    }, config=cfg)
    print("Tool calls:", result["tool_calls_made"])
    print("Papers found:", len(result["papers_found"]))
    print(result["messages"][-1].content)
    for p in result["papers_found"][:3]:
        title = p.get("paper_title") or p.get("title") or "(no title)"
        print(" -", title[:80])

    result = app.invoke({
        "messages": [HumanMessage(content="How does it compare with full fine-tuning?")],
    }, config=cfg)
    print("Tool calls:", result["tool_calls_made"])
    print("Papers found:", len(result["papers_found"]))
    print(result["messages"][-1].content)
    for p in result["papers_found"][:3]:
        title = p.get("paper_title") or p.get("title") or "(no title)"
        print(" -", title[:80])
    

    