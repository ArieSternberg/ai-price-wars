"""Render the LLM agent's LangGraph topology as Mermaid — for the README.

The graph structure itself (node names, edges) doesn't depend on any particular
round or model, so this builds a placeholder graph with the same shape as
`LLMVendor.decide_price` in pricewars/agents/llm.py, purely to extract the diagram.
No API calls.

Usage:
    python scripts/render_agent_graph.py
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from pricewars.agents.llm import AgentState

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "figures" / "agent_graph.mmd"


def build_placeholder_graph() -> StateGraph:
    """Same node names and edges as LLMVendor.decide_price — see agents/llm.py."""

    def agent_node(state: AgentState) -> dict:
        raise NotImplementedError("placeholder — for diagram extraction only")

    def tools_node(state: AgentState) -> dict:
        raise NotImplementedError("placeholder — for diagram extraction only")

    def route_after_agent(state: AgentState) -> str:
        return "tools"

    def route_after_tools(state: AgentState) -> str:
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return graph


def main() -> None:
    compiled = build_placeholder_graph().compile()
    mermaid = compiled.get_graph().draw_mermaid()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(mermaid)
    print(mermaid)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
