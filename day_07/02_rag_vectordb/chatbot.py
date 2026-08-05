from os import getenv
from pathlib import Path
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from system_prompt import SYSTEM_PROMPT

load_dotenv()

OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = getenv("OPENROUTER_BASE_URL")
MODEL = getenv("MODEL")
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL")

DOCUMENT_URL = "https://arxiv.org/pdf/2510.26692"
DOCUMENT_PATH = Path(__file__).parent / "kimi.pdf"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = f"kimi-{EMBEDDING_MODEL}"

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found. Add it to your .env file.")

API_KEY = SecretStr(OPENROUTER_API_KEY)

# step 1: download the PDF document
def download_pdf(url, file_path):
    if file_path.exists():
        print(f"Using cached PDF: {file_path.name}")
        return

    print(f"Downloading {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    file_path.write_bytes(response.content)
    print(f"Saved {file_path.name} ({len(response.content) / 1_000_000:.1f} MB)")

download_pdf(DOCUMENT_URL, DOCUMENT_PATH)