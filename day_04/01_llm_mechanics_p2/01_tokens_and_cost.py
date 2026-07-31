import os
import requests
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ["MODEL"]

# 1. Get the model catalogue and pricing information from OpenRouter
catalogue = requests.get("https://openrouter.ai/api/v1/models", timeout=10).json()["data"]
pricing = next((m["pricing"] for m in catalogue if m["id"] == MODEL), None)
if pricing is None:
    raise SystemExit(f"Model {MODEL} not found in catalogue.")
USD_PER_M_IN = float(pricing["prompt"]) * 1_000_000
USD_PER_M_OUT = float(pricing["completion"]) * 1_000_000
USD_INR = requests.get("https://api.frankfurter.app/latest?from=USD&to=INR", timeout=10).json()["rates"]["INR"]
print(f"{MODEL} ${USD_PER_M_IN:.4f}/M in . ${USD_PER_M_OUT:.4f}/M out. 1 USD = {USD_INR:.2f} INR \n")

def cost(in_tok: int, out_tok: int) -> float:
    return in_tok / 1_000_000 * USD_PER_M_IN + out_tok / 1_000_000 * USD_PER_M_OUT

def price(in_tok: int, out_tok: int) -> str:
    usd = cost(in_tok, out_tok)
    return f"${usd:.6f} (₹{usd * USD_INR:.4f})"

POLICY = ("Sheldon Retail Bank offers a range of financial services including savings accounts, loans, and investment options."
        "Our mission is to provide exceptional customer service and innovative solutions to help our clients achieve their financial goals."
        "We are committed to maintaining the highest standards of integrity and transparency in all our operations.")

prompt = f"Summarize the policy text in three numbered points:\n\n{POLICY}"

# 2. Estimate the number of tokens in the prompt and the expected number of tokens in the output
est_in = len(tiktoken.get_encoding("cl100k_base").encode(prompt))
est_out = 100
print(f"Estimated tokens: {est_in} in, {est_out} out. Estimated cost: {price(est_in, est_out)}\n")

# 3. Actually call the model and get the response
client = OpenAI(base_url=os.environ["OPENROUTER_BASE_URL"], api_key=os.environ["OPENROUTER_API_KEY"])
response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
print(f"Model response:\n{response.choices[0].message.content}\n")

# 4. Verify what it really cost by counting the tokens in the prompt and the response
usage = response.usage
if usage:
    print(f"Actual tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out. Actual cost: {price(usage.prompt_tokens, usage.completion_tokens)}")

    # 5. Scale that one call to 10_000 calls and see what the total cost would be
    CALLS_PER_DAY = 10_000
    DAYS = 30
    monthly = cost(usage.prompt_tokens, usage.completion_tokens) * CALLS_PER_DAY * DAYS
    print(f"at {CALLS_PER_DAY} calls/day for {DAYS} days, the total cost would be: ${monthly:.2f} (₹{monthly * USD_INR:.2f})")

