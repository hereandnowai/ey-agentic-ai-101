import os
import time
from openai import (
    OpenAI,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
    APIError,
)
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
MODEL = os.environ["MODEL"]
QUESTION: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "One sentence: why keep API keys out of source code?"}
]

client_ok = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=20,
    max_retries=0
)

client_impatient = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=0.001,
    max_retries=0
)

client_bad_key = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key="sk-or-v1-DELIBERATELY-INVALID-KEY",
    timeout=20.0,
    max_retries=0
)

TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
TOO_BROAD = (RateLimitError, APITimeoutError, APIError)

print("AuthenticationError is a subclass of APIError:", issubclass(AuthenticationError, APIError))

def chat_with_retries(client: OpenAI,
                      messages: list[ChatCompletionMessageParam],
                      retry_on: tuple[type[Exception], ...] = TRANSIENT,
                      max_attempts: int = 4) -> str:
    """Call the OpenAI chat API with retries for transient errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0)
            return resp.choices[0].message.content or ""
        except retry_on as error:
            if attempt == max_attempts:
                raise
            wait = 0.5 * (2 ** (attempt - 1))  # Exponential backoff
            print(f" [retry] attempt {attempt} failed ({type(error).__name__});"
                  f"waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Unreachable code reached in chat_with_retries")

def run(label: str, **kwargs) -> None:
    """Run one demo and print how long it took, whether it worked or not"""
    print(f"\n=== {label} ===")
    started = time.time()
    try:
        answer = chat_with_retries(**kwargs)
        print("Answer:", answer)
    except Exception as error:
        print(f" gave up: {type(error).__name__}")
    print(f" elapsed: {time.time() - started:.2f}s")

run("DEMO 1: healthy call (expect no retries)", client=client_ok, messages=QUESTION)

run("DEMO 2: impatient call (expect retries)", client=client_impatient, messages=QUESTION)

run("DEMO 3: bad key with retries (expect retries) ", client=client_bad_key, messages=QUESTION, retry_on=TOO_BROAD)

run("DEMO 4: bad key (expect no retries)", client=client_bad_key, messages=QUESTION, retry_on=TRANSIENT)