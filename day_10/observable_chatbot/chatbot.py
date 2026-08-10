import asyncio
import json
import os
import urllib.request
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, Message
from agent_framework.observability import enable_instrumentation
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from langfuse import Langfuse
from pydantic import Field

load_dotenv()
MODEL = os.environ["MODEL"]
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = os.environ["OPENROUTER_BASE_URL"]

# 1. OBSERVABILITY: Langfuse creates the OpenTelemetry pipeline.
langfuse = Langfuse()
enable_instrumentation(enable_sensitive_data=True)

# 2. PRICE: fetching the cost of an llm from openrouter
with urllib.request.urlopen("https://openrouter.ai/api/v1/models") as response:
    PRICES = {m["id"]: m["pricing"] for m in json.load(response)["data"]}

PRICE_IN = float(PRICES[MODEL]["prompt"])
PRICE_OUT = float(PRICES[MODEL]["completion"])
print(f"[obs] {MODEL} ${PRICE_IN * 1e6:.2f}/M in, ${PRICE_OUT * 1e6:.2f}/M out")

# 3. THE CHATBOT. One tool, so a turn makes more than one model call
ACCOUNTS = {"SB-9001": 84_215.50, "SB-9002": 12_430.00, "SB-9003": 3_46_890.25}

def check_balance(account_id: Annotated[str, Field(description="Account id, e.g. SB-9001")]) -> str:
    """Look up the balance of a Meridian Bank Account"""
    balance = ACCOUNTS.get(account_id.upper())
    return f"{account_id}: Rs {balance:,.2f}" if balance else f"No account {account_id} found"

agent = Agent(OpenAIChatCompletionClient(model=MODEL, api_key=API_KEY, base_url=BASE_URL),
              "You are Meridian Bank's Assistant. Branches open Mon-Fri 9:30-16:30, "
              "Sat 9:30-13:30. Savings pays 2.75%, fixed deposits 7.25%. Use the tool for "
              "any balance related question - never guess one. Be brief.",
              name="MeridianAssist",
              tools=[check_balance]
              )

# 4. ONE TURN = ONE TRACE.
def ask(message: str, history: list) -> tuple[str, str]:
    past = [Message(m["role"], [m["content"]]) for m in history]

    with langfuse.start_as_current_observation(name="chat turn", as_type="agent") as span:
        result = asyncio.run(agent.run([*past, Message("user", [message])])) # type: ignore
        trace_id = span.trace_id

    langfuse.flush()

    used = result.usage_details or {}
    tokens_in = used.get("input_token_count") or 0
    tokens_out = used.get("output_token_count") or 0
    cost = tokens_in * PRICE_IN + tokens_out * PRICE_OUT

    report = (f"**{tokens_in:,}** tokens in **{tokens_out:,}** out **${cost:.6f}**\n\n"
              f"[See every step in Langfuse]({langfuse.get_trace_url(trace_id=trace_id)})")
    return result.text, report