import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL = os.environ["MODEL"]

# 1. Declare the expected structured output format using a Pydantic model
class LoanEnquiry(BaseModel):
    intent: str = Field(description="e.g. apply, enquire, complain")
    product: str
    amount: float | None = Field(default=None, description="loan amount if stated, else null")
    tenure_years: int | None = None
    applicant_name: str | None = None

system_prompt = """Extract loan enquiry into the given schema.
               Use null for anything not stated. Amounts in rupees
               (50 lakh = 5000000). Return only the JSON object, no extra text."""

message = ("""Hi, I'am Sheldon Cooper. I'd like to apply for a home loan of about 50 lakh rupees
           for a tenure of 20 years. Can you help me with the process?""")

# 2. Request the model to return structured output in the defined format
completion = client.beta.chat.completions.parse(
    model=MODEL,
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": message}],
    response_format=LoanEnquiry
)

# 3. Access the structured output as a Pydantic model instance
enquiry = completion.choices[0].message.parsed
if enquiry is None:
    raise RuntimeError("Model returned no parsed object (refusal or truncation).")
print("Parsed structured output :", enquiry)
print("Product                  :", enquiry.product)
print("Amount                   :", enquiry.amount)
print("Parsed structured output :", enquiry.model_dump_json(indent=2))