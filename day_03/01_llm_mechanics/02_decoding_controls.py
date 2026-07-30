import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL = os.environ["MODEL"]

PROMPT = """Give a name for a new savings account product. Reply with just the name!"""

def generate(temperature: float, max_tokens: int = 20, stop=None) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=stop
    )
    return (resp.choices[0].message.content or "").strip()

print("=== temperature 0.0 (run 3x - expect near-identical) ===")
for _ in range(3):
    print(" ", generate(0.0))

print("\n=== temperature 1.0 (run 3x - expect variety) ===")
for _ in range(3):
    print(" ", generate(1.0))

print("\n=== max_tokens=2 (truncates)")
print(" ", generate(temperature=0.7, max_tokens=2))