from dotenv import load_dotenv
from os import getenv
import os
import urllib.request
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from system_prompt import SYSTEM_PROMPT

load_dotenv()
MODEL = os.environ["MODEL"]

llm = init_chat_model(
    model=MODEL,
    model_provider="openrouter",
    temperature=0)

PDF_PATH = os.path.join(os.path.dirname(__file__), "kimi.pdf")
PDF_URL = "https://arxiv.org/pdf/2510.26692"

def download_pdf(url, file_path):
    if os.path.exists(file_path):
        return
    with urllib.request.urlopen(url, timeout=10) as response:
        content = response.read()
    with open(file_path, "wb") as f:
        f.write(content)

download_pdf(PDF_URL, PDF_PATH)

def load_pdf_context(file_path):
    if not os.path.exists(file_path):
        return f"Warning: {file_path} not found. Proceeding without the document context."
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return "\n".join([page.page_content for page in pages])

pdf_context = load_pdf_context(PDF_PATH)
full_system_prompt = f"{SYSTEM_PROMPT}\n\nContext from PDF:\n{pdf_context}"

messages: list[BaseMessage] = [SystemMessage(content=full_system_prompt)]

def get_streaming_response(user_input):
    global messages
    messages.append(HumanMessage(content=user_input))

    full_response = ""
    for chunk in llm.stream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            full_response += content
            yield ("response", content)

    messages.append(AIMessage(content=full_response))