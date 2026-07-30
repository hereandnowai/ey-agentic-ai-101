# system = standing instructions for the model, like "You are a helpful assistant"
# user = the human's input, like "What is the capital of France?"
# assistant = the model's output, like "The capital of France is Paris."

import os
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL = os.environ["MODEL"]

def ask(messages: list[ChatCompletionMessageParam]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content or ""

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "you are a banking assistant. Answer ONLY in a few lines. If ask for legal or tax advice decline politely"},
    {"role": "user", "content": "What documents do I need for a home loan?"}
]
answer1 = ask(messages)
print("assistant: ", answer1)

messages.append({"role": "assistant", "content": answer1})
messages.append({"role": "user", "content": "And is the interest fixed or floating?"})
answer2 = ask(messages)
print("assistant: ", answer2)

messages.append({"role": "assistant", "content": answer2})
messages.append({"role": "user", "content": "Give me tax advice to reduce my liability."})
print("\nassistant: ", ask(messages))
