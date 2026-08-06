# what is an edge in langgraph?
# an edge is a connection between two nodes in a graph. It represents the flow of information
# from one node to another. An edge can also have a reducer that combines the output of
# the source node with the input of the target node.

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    steps: Annotated[list, add]

def intake(state):
    return {"steps": ["intake"]}

def price(state):
    return {"steps": ["price"]}

def notify(state):
    return {"steps": ["notify"]}

b = StateGraph(S)
b.add_node("intake", intake)
b.add_node("price", price)
b.add_node("notify", notify)
b.add_edge(START, "intake")
b.add_edge("intake", "price")
b.add_edge("price", "notify")
b.add_edge("notify", END)
graph = b.compile()

print(graph.invoke({"steps": []})["steps"])