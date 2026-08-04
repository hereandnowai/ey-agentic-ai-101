from dotenv import load_dotenv
import os
import urllib.request
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from system_prompt import SYSTEM_PROMPT

load_dotenv()
MODEL = os.environ["MODEL"]

llm = init_chat_model(
    model=MODEL,
    model_provider="openrouter",
    temperature=0)

DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "genoffice.md")
DOCUMENT_URL = "https://raw.githubusercontent.com/genspark-ai/genoffice/refs/heads/main/README.md"

def download_document(url, file_path):
    if os.path.exists(file_path):
        return
    with urllib.request.urlopen(url, timeout=10) as response:
        content = response.read().decode("utf-8")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

download_document(DOCUMENT_URL, DOCUMENT_PATH)

def load_text_context(file_path):
    if not os.path.exists(file_path):
        return f"Warning: {file_path} not found. Proceeding without the document context."
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

document_context = load_text_context(DOCUMENT_PATH)
full_system_prompt = f"{SYSTEM_PROMPT}\n\nContext from document:\n{document_context}"

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