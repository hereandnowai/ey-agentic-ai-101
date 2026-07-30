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

messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": "You are a helpful assistant"}]

print("=== Chatbot with memory ===")
print("Type 'quit' to exit.\n")
while True:
    user_input = input("you: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"quit", "exit"}:
        print(f"Goodbye!")
        break

    # 1. Add the human's new question to the END of the transcript.
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    reply = response.choices[0].message.content or ""

    messages.append({"role": "assistant", "content": reply})
    print(f"Caramel AI: {reply}\n")
















    # if user_input.lower() == "history":
    #     print("\n--- what we sent to the model every turn ---")
    #     for msg in messages:
    #         print(f" {msg['role']:>9}: {str(msg.get('content'))[:100]}")
    #     print(f"--- {len(messages)} messages total ---\n")
    #     continue

