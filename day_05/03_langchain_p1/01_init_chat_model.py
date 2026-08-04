import os
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ["MODEL"]

llm = init_chat_model(MODEL, model_provider="openrouter", temperature=0)

response = llm.invoke([
    SystemMessage("You are a concise banking assistant."),
    HumanMessage("In one sentence, what is a floating interest rate?")
])

print("type.        :", type(response).__name__)
print("content      :", response.content)
print("tokens.      :", response.usage_metadata)
print("\n streaming :", end="", flush=True)
for chunk in llm.stream([HumanMessage("List two types of bank accounts, briefly.")]):
    print(chunk.content, end="", flush=True)
print()