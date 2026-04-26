from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import settings
from src.state import AgentState
from src.agents.prompts import ANALYSIS_SYSTEM

llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_chat_model,
    temperature=0.2, # un poco mas de creatividad
)

def format_papers(papers: list[dict]):
    if not papers:
        return "(no papers found)"
    lines = []
    for p in papers[:8]:
        title = p.get("paper_title") or p.get("title", "Unknown")
        pid = p.get("paper_id", "?")
        body = p.get("passage") or p.get("abstract", "")
        lines.append(f"--- [{title}, id={pid}] ---\n{body[:800]}\n")
    return "\n".join(lines)

def analysis_node(state: AgentState) -> dict:
    # Find the original user question (first HummanMessage)
    user_q = "(no question)"
    for m in state["messages"]:
        if m.type == "human":
            user_q = m.content; break

    context = format_papers(state.get("papers_found", []))
    prompt = f"""Users question: {user_q}
    Papers retrieved by Reasearch Agent:
        
    {context}
      
    Now answer the user's question using ONLY these papers. Cite inline."""

    response = llm.invoke([
        SystemMessage(content=ANALYSIS_SYSTEM),
        HumanMessage(content=prompt)
    ])
    return {"messages": [response]}

    