# what is a node in the context of programming?
# a node is a single unit of work that can be executed independently.

# what is a node in the context of langgraph?
# a node is one step of the graph: a plain function that receives the shared state
# and returns a new state.

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

class App(TypedDict):
    message: str
    intent: NotRequired[str]
    reply: NotRequired[str]

class Intent(BaseModel):
    intent: Literal["apply", "complain", "enquire"]

def classify(state):
    result = cast(Intent, model.with_structured_output(Intent).invoke(
        "Classify the customer's intent: " + state["message"]
    ))
    return {"intent": result.intent}

def respond(state):
    replies = {
        "apply": "Starting your application.",
        "complain": "Logging your complain.",
        "enquire": "Here is the information"
    }
    return {"reply": replies[state["intent"]]}

b = StateGraph(App)
b.add_node("classify", classify)
b.add_node("respond", respond)
b.add_edge(START, "classify")
b.add_edge("classify", "respond")
b.add_edge("respond", END)
graph = b.compile()

print(graph.invoke({"message": "I want to open a home loan"}))