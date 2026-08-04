import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import cast

load_dotenv()
MODEL = os.environ["MODEL"]
llm = init_chat_model(MODEL, model_provider="openrouter", temperature=0)

# 1. a prompt template with {placeholder} for user input
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Meridian Bank's assistant. Answer in one sentence."),
    ("user", "Explain '{term}' to a {audience}.")
])

# 2. compose a chain with the pipe operator
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"term": "EMI", "audience": "first-time borrower"}))

# 3. structured output: validated pydantic straight out of the model
class LoanEnquiry(BaseModel):
    intent:  str
    product: str
    amount: float | None = Field(default=None, description="rupees; null if unstated")

structured_model = llm.with_structured_output(LoanEnquiry)
result = cast(LoanEnquiry, structured_model.invoke(
    "I'd like to apply for a home loan of about 50 lakhs."
))
print("\nstructured: ", result)
print("product.    : ", result.product, "| amount: ", result.amount)

extract_chain = ChatPromptTemplate.from_messages([
    ("system", "Extract the enquiry. Amounts in rupees (1 lakh = 100000). Null if unstated."),
    ("user", "{message}")
]) | structured_model
print("\nchained : ", extract_chain.invoke({"message": "Enquiring about a 30 lakh home loan"}))