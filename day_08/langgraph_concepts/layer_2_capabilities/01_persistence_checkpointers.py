# What is checkpointers in langgraph?
# A checkpointer is MEMORY for a graph.

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()
model = init_chat_model(
    os.environ["MODEL"],
    model_provider="openrouter",
    temperature=0
)

def chat(state):
    return {"messages": [model.invoke(state["messages"])]}

b = StateGraph(MessagesState)
b.add_node("chat", chat)
b.add_edge(START, "chat")
b.add_edge("chat", END)
graph = b.compile(checkpointer=InMemorySaver())

penny: RunnableConfig = {"configurable": {"thread_id": "penny"}}
graph.invoke({"messages": [HumanMessage(content="Hi, my name is Penny.")]}, penny)
answer = graph.invoke({"messages": [HumanMessage(content="What us my name?")]}, penny)
print("same thread ->", answer["messages"][-1].content)