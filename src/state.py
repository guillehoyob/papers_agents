from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    # Poblado por Research Agent
    papers_found: list[dict] # [{paper_id, title, passages, source}]

    # Control de flujo
    research_complete: bool

    # Para debug / Observabilidad
    tool_calls_made: list[str]