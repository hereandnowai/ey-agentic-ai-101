import os
from typing import TypedDict, NotRequired, Literal, cast
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

load_dotenv()
model = init_chat_model(
    os.environ["MODEL"],
    model_provider="openrouter",
    temperature=0
)

class S(TypedDict):
    request: str
    decision: NotRequired[str]
    reason: NotRequired[str]

class Verdict(BaseModel):
    decision: Literal["approve", "decline"]
    reason: str

def triage(state) -> Command[Literal["approve", "decline"]]:
    verdict = cast(Verdict, model.with_structured_output(Verdict).invoke(
        "Approve or decline this loan, with a short reason:\n" + state["request"]
    ))
    return Command(update=verdict.model_dump(), goto=verdict.decision)

def approve(state):
    return {}

def decline(state):
    return {}

b = StateGraph(S)
b.add_node("triage", triage)
b.add_node("approve", approve)
b.add_node("decline", decline)
b.add_edge(START, "triage")
b.add_edge("approve", END)
b.add_edge("decline", END)
graph = b.compile()

print(graph.invoke({"request": "Home loan of 30 lakhs, income 2 lakhs/month, CIBIL 780."}))