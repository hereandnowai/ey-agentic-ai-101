import os
import math
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

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(theta) = (a . b) / (|a| * |b|). Pure python - no numpy needed"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

passages = [
    "Meridian home loans require a minimum CIBIL score of 700.",
    "Savings accounts earn 3.0% interest per annum.",
    "Report a lost of stolen card immediately via the app."
]
query = "What credit score do I need to borrow for a house?"

passages_vecs = embeddings.embed_documents(passages)
query_vec = embeddings.embed_query(query)
scored = [(cosine_similarity(query_vec, pv), p) for pv, p in zip(passages_vecs, passages)]
scored.sort(reverse=True)

print(f"query: {query}\n")
for score, passage in scored:
    print(f" {score:.4f} {passage}")

print(f"\n best match -> {scored[0][1]}")