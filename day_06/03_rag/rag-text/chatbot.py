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

DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "profile-rr.md")
DOCUMENT_URL = "https://raw.githubusercontent.com/hereandnowai/genai-and-prompt-engineering-eduhubspot-s1/refs/heads/main/day-6-of-14/6-chatbot-with-text/profile-rr.md"
