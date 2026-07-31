import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL = os.environ["MODEL"]

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Roast python developers in 20 lines!"}],
    stream=True
)

print("streaming: ", end="", flush=True)
collected = []
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
        collected.append(delta)

full_text = "".join(collected)
print("\n\nfull length: ", len(full_text), "chars")
