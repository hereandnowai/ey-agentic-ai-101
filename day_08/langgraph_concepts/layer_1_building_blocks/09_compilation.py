from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    n: int

def double(state):
    return {"n": state["n"] * 2}

builder = StateGraph(S)
builder.add_node("double", double)
builder.add_edge(START, "double")
builder.add_edge("double", END)

print("before compilation:", type(builder).__name__)
graph = builder.compile()
print("after compilation:", type(graph).__name__)

print(graph.invoke({"n": 21}))
print(graph.invoke({"n": 42}))