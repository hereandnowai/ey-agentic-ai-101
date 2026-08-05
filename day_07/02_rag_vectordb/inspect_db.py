import sqlite3
from os import getenv
from dotenv import load_dotenv
from pathlib import Path
import chromadb
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = getenv("OPENROUTER_BASE_URL", "https://api.openrouter.ai/v1")

if not CHROMA_PATH.exists():
    raise SystemExit(f"Chroma DB path {CHROMA_PATH} does not exist. Run app.py to create it first.")

def heading(text):
    print(f"\n{'=' * 80}\n{text}\n{'=' * 80}\n")

def got(result, key):
    """One column from collection.get() - a flat list of values for a single key."""
    values = result.get(key)
    return [] if values is None else list(values)

def found(result, key):
    """One column from collection.query() - a list holding one list per question."""
    values = result.get(key)
    return [] if values is None else list(values[0])

# --- 1. what is actually on disk? ---
heading("1. Inspecting the Chroma DB on disk")
for path in sorted(CHROMA_PATH.rglob("*")):
    if path.is_file():
        print(f" {path.stat().st_size / 1024:>9,.0f} KB {path.relative_to(CHROMA_PATH)}")

print("\n chroma.sqlite3 -> an ordinary database file, the text and its labels")
print(" <uuid>/*.bin -> the raw numbers, packed together for fast retrieval")

# --- 2. The collection ---
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
names = [col.name for col in client.list_collections()]

heading("2. What is inside the Chroma DB collection?")
for name in names:
    print(f" collection: {name} holds {client.get_collection(name).count()} items")

print("\n The collection NAME is not a folder name, it is recorded inside")
print(" chroma.sqlite3, the folder is named after the vector segment.")

connection = sqlite3.connect(CHROMA_PATH / "chroma.sqlite3")
mapping = connection.execute(
    "select c.name, s.id from collections c join segments s on s.collection = c.id"
    " where s.scope = 'VECTOR'"
).fetchall()
connection.close()
for name, folder in mapping:
    print(f" collection: {name} -> folder: {folder}")
collection = client.get_collection(names[0])

# --- 3. One chunk, up close ---
sample = collection.get(limit=1, include=["metadatas", "documents", "embeddings"])
document = got(sample, "documents")[0]
metadata = got(sample, "metadatas")[0]
vector = got(sample, "embeddings")[0]

heading("3. Inspecting one chunk of text, its metadata, and its vector")
print(f" Its id        : {got(sample, 'ids')[0]}")
print(f" from PDF page : {metadata.get('page')}")
print(f" text length   : {len(document)} characters")
print(f" text it sorted: \n {' '.join(document.split())}")