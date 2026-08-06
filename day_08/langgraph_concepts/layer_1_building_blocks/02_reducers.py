# what is a reducer in the context of programming?
# a reducer is just a rule for combining two things into one.

# what is reducer in the context of langgraph?
# a reducer is a rule for combining two nodes into one node.

from typing import TypedDict, Annotated
from operator import add 
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    audit: Annotated[list, add]
    status: str

def receive(state):
    return {"audit": ["received"], "status": "in_review"}

def decide(state):
    return {"audit": ["approved"], "status": "approved"}

b = StateGraph(S)
b.add_node("receive", receive)
b.add_node("decide", decide)
b.add_edge(START, "receive")
b.add_edge("receive", "decide")
b.add_edge("decide", END)
graph = b.compile()

print(graph.invoke({"audit": ["created"], "status": "new"}))