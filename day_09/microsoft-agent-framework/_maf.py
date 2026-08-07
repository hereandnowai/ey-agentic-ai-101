import asyncio
import os
from typing import Any
from dotenv import find_dotenv, load_dotenv
from agent_framework_openai import OpenAIChatCompletionClient

load_dotenv(find_dotenv())

BASE_URL = "hthps://openrouter.ai/api/v1"
BACKEND = "openrouter"
MODEL = "openai/gpt-5.6-luna"

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    raise SystemExit("Please set the OPENROUTER_API_KEY environment variable.")

def get_client(**overrides: Any) -> OpenAIChatCompletionClient:
    """One chat client, pointed at OpenRouter"""
    settings: dict[str, Any] = dict(model=MODEL, api_key=API_KEY, base_url=BASE_URL)
    settings.update(overrides)
    return OpenAIChatCompletionClient(**settings)

def banner(title: str) -> None:
    """Every file opens with this, so the room knows what served the run"""
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f" backend: {BACKEND}   model: {MODEL}")

def run(coro):
    """Run as async main()" from a plain script"""
    return asyncio.run(coro)


POLICY = ("Sheldon Retail accepts audio returns within 21 days of delivery. "
          "Faulty goods carry a two-year warranty. Faulty returns are free.")
ORDERS = {"SC-90455": {"days": 12, "faulty": True, "item": "Sheldon Studio Headphones"},
          "SC-90456": {"days": 34, "faulty": False, "item": "Sheldon Buds Mk II"}}