import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import cast
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL = os.environ["MODEL"]

# 1. Define the tool to be called by the model
def calculate_emi(principal: float, annual_rate: float, years: int) -> dict:
    """Computer the EMI (Equated Monthly Installment) for a loan."""
    r = annual_rate / 12 / 100  # monthly interest rate
    n = years * 12  # total number of monthly payments
    if r == 0:
        emi = principal / n
    else:
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n -1)
    return {"emi": round(emi, 2), "total_payable": round(emi * n, 2)}

# 2. Define the tool schema for the model
tools: list[ChatCompletionToolParam] = [{
    "type": "function",
    "function": {
        "name": "calculate_emi",
        "description": "Computer the EMI (Equated Monthly Installment) for a loan.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "loan amount in rupees"},
                "annual_rate": {"type": "number", "description": "annual interest rate percent"},
                "years": {"type": "integer", "description": "loan tenure in years"},
            },
            "required": ["principal", "annual_rate", "years"],
        },
    },
}]

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "I want to take a home loan of 50 lakh rupees at 7% annual interest for 20 years. Can you calculate the EMI for me?"
    }
]

# 3. First Request to the model to call the tool and return the result
first = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
msg = first.choices[0].message
messages.append(cast(ChatCompletionMessageParam, msg))

if msg.tool_calls:
    for tc in msg.tool_calls:
        if tc.type != "function":
            continue
        args = json.loads(tc.function.arguments)
        print("model want to call: ", tc.function.name, "with", args)
        result = calculate_emi(**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result)
            }
        )

# 4. second call: the model turns the tool result into a natural answer.
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
print("Final answer from model: ", final.choices[0].message.content)

