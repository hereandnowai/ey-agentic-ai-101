import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.messages import AIMessage

load_dotenv()
MODEL = os.environ["MODEL"]
model = init_chat_model(MODEL, model_provider="openrouter", temperature=0)

# What is a tool? | Tool is just a typed, docstringed function with @tool
@tool
def calculate_emi(principal: float, annual_rate: float, years: int) -> dict:
    """Calculate the monthly EMI and total payable for a loan.
    
    Args:
        principal: loan amount in rupees.
        annual_rate: annual interest rate as a percent (e.g. 8.4).
        years: loan tenure in years.        
    """
    r = annual_rate / 12 / 100  # monthly interest rate
    n = years * 12  # total number of monthly payments
    if r == 0:
        emi = principal / n
    else:
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n -1)
    return {"emi": round(emi, 2), "total_payable": round(emi * n, 2)}

# build the agent = llm + tools
agent = create_agent(
    model=model,
    tools=[calculate_emi],
    system_prompt="You a Meridian Bank's Assistant. Use tools for any calculation."
)
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the EMI on a 50 lakh loan at 8.4% over 25 years?"
    }]
})

print("=== step trace ===")
for m in result["messages"]:
    kind = m.type
    if isinstance(m, AIMessage) and m.tool_calls:
        print(f"[{kind}] tool_call -> {m.tool_calls[0]['name']}{m.tool_calls[0]['args']}")
    elif m.content:
        print(f"[{kind}] {m.content}")

print("\nfinal answer:", result["messages"][-1].content)