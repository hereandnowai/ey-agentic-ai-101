import os
from langchain_openai import OpenAIEmbeddings
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

vec = embeddings.embed_query("home loan eligibility")
print("text       : 'home loan eligibility'")
print("vector dim :", len(vec))
print("first 5 dim:", [round(x, 4) for x in vec[:5]])

docs = ["saving account interest", "how to block a stolen card"]
doc_vecs = embeddings.embed_documents(docs)
print("\nbatched embeddings: ", len(doc_vecs),  "vectors, each dim", len(doc_vecs[0]))