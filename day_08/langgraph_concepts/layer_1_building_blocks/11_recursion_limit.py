# what is recursion limit?
# A recursion limit is a safety cap: "don't repeat more than this many times."
# It is a limit on how many times a function can call itself before it stops
# and raises an error. This is important to prevent infinite loops and
# stack overflow errors in programming.

# what is recursion limit in langgraph?
# Recursion limit in langgraph is a limit on how many times a node can call itself.


from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

class S(TypedDict):
    count: int

def tick(state):
    return {"count": state["count"] + 1}

def loop_back(state):
    return "tick"

b = StateGraph(S)
b.add_node("tick", tick)
b.add_edge(START, "tick")
b.add_conditional_edges("tick", loop_back, ["tick"])
graph = b.compile()

try:
    graph.invoke({"count": 0}, {"recursion_limit": 5})
except GraphRecursionError:
    print("Stopped by recursion limit=5 - the seatbeld worked, no infinite loop.")