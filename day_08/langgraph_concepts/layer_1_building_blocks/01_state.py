# langchain -> for building an LLM app or an agent.
# langgraph -> for building multi-agent systems and multi-step reasoning workflows.

# what is state?
# a state is a snapshot of the current situation in a multi-step reasoning workflow.
# It contains all the information needed to make decisions and take actions.

# what is a state in langgraph?
# State is one shared box of information.

from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END

class LoanState(TypedDict):
    applicant: str
    income: float
    emi: float
    ratio: NotRequired[float]

def compute_ratio(state):
    return {"ratio": round(state["emi"] / state["income"], 2)}


b = StateGraph(LoanState)
b.add_node("compute_ratio", compute_ratio)
b.add_edge(START, "compute_ratio")
b.add_edge("compute_ratio", END)
graph = b.compile()

print(graph.invoke({"applicant": "Lea", "income": 150000, "emi": 45000}))