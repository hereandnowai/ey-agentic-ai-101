# Objective of this file:
# we are going to ask same kind of question three different ways

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = os.environ["MODEL"]

def ask(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    return resp.choices[0].message.content or ""

# 1. Role framing - set who the model is and the rules of the game.
print("1. Role framing")
print(ask(
    system="You are Missy Bank support agent. Answer in one crisp sentence.",
    user="Do you charge for UPI transfers?"
))

# 2. FEW-SHOT - teach the desired format by example. Great for consistent structure.
print("\n2. Few-shot (classification)")
few_shot = (
    "Classify the customer message intent as one of: apply, complain, enquire.\n"
    "Message: 'My card was charged twice' -> complain\n"
    "Message: 'What is the FD rate?' -> enquire\n"
    "Message: 'I want to open a savings account' -> apply\n"
    "Reply with only the label."
)
print(ask(system=few_shot,
          user="Message: 'How do I apply for a home loan?'"))

# 3. Chain-of-thought - ask the model to reason step by step before answering.
print("\n3. Chain-of-thought")
print(ask(
    system="Decide loan eligibility. Think step-by-step the rule then end with "
          "'Decision: <approve/refer/decline>'. Rule: apprive if CIBIL>=750 and EMI/income<=40%.",
    user="Customer has CIBIL 780 and EMI 35% of income. Is he eligible for a loan?"
))