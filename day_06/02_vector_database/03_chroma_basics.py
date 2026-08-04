import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
BASE_URL = os.environ["OPENROUTER_BASE_URL"]
API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    check_embedding_ctx_length=False
)

docs = [
    Document(page_content="Home loans require a minimum CIBIL score of 700.",
             metadata={"id": "hl-001", "category": "home_loan"}),
    Document(page_content="Home loans rates start from 8.4% for a scores of 750+.",
             metadata={"id": "hl-002", "category": "home_loan"}),
    Document(page_content="Savings accounts earn 3.0% interest per annum.",
            metadata={"id": "sa-001", "category": "savings_account"}),
    Document(page_content="Block a lost or stolen card immediately via the app.",
             metadata={"id": "cc-001", "category": "credit_card"}),
             
]

store = Chroma(
    collection_name="meridian_demo",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
store.add_documents(docs)

# Similarity Search
print("=== plain search: 'credit score to buy a house' ===")
for d in store.similarity_search("credit score to buy a house", k=2):
    print(f" [{d.metadata['category']}] {d.page_content}")

# Metadata filtering


# with scores