# cosine similarity measures the angle between two vectors,
# which can be used to determine how similar they are.
# It is commonly used in natural language processing and
# information retrieval to compare text documents or embeddings.

# 1.0 = same direction (most similar / identical vectors)
# 0.0 = unrelated orthogonal (no similarity)

# We rank passages against a query by its cosine similarity score,
# which is the dot product of the two vectors divided by the product of their magnitudes.

import os
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
)
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]

def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

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

query_vec = embed(query)
scored = [(cosine_similarity(query_vec, embed(p)), p) for p in passages]
scored.sort(reverse=True)

print(f"query: {query}\n")
for score, passage in scored:
    print(f" {score:.4f} {passage}")

print(f"\n best match -> {scored[0][1]}")