import os
from typing import TypedDict, NotRequired, Literal, cast
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

load_dotenv()
model = init_chat_model(
    os.environ["MODEL"],
    model_provider="openrouter",
    temperature=0
)

class Triage(TypedDict):
    message: str
    category: NotRequired[str]
    reply: NotRequired[str]

class Cat(BaseModel):
    category: Literal["loan", "card", "fraud"]

def classify(state):
    result = cast(Cat, model.with_structured_output(Cat).invoke(
        "Classify the customer's message: " + state["message"]
    ))
    return {"category": result.category}

def make_specialist(area):
    def specialist(state):
        answer = model.invoke(
            f"You are Meridian Bank's {area} specialist. Answer in one sentence.\n"
            + state["message"]
        )
        return {"reply": answer.content}
    return specialist

def route(state):
    return state["category"]

b = StateGraph(Triage)
b.add_node("classify", classify)
for area in ["loan", "card", "fraud"]:
    b.add_node(area, make_specialist(area))
b.add_edge(START, "classify")
b.add_conditional_edges("classify", route, ["loan", "card", 'fraud'])
for area in ["loan", "card", "fraud"]:
    b.add_edge(area, END)
agent = b.compile()

for msg in ["What documents do I need for a home loan?", "Someone used my card abroad", "I was double charged"]:
    out = agent.invoke({"message": msg})
    print(f"[{out['category']}] {out['reply']}")