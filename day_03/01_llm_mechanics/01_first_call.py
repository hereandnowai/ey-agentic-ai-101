import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL = os.environ["MODEL"]

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a banking assistant"},
        {"role": "user", "content":"Write a funny story on EMI in 200 words!"} # $5 per 1M tokens
    ]
)

print("reply: ", response.choices[0].message.content) # $30 per 1M tokens

usage = response.usage
if usage is not None:
    print("Input Tokens : ", usage.prompt_tokens)
    print("Output Tokens: ", usage.completion_tokens)
    print("Total Tokens : ", usage.total_tokens)
else:
    print("usage        : ", "not reported by the provider")