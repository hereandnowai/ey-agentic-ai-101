# Embeddings basics
# An embedding is a numerical representation of an object,
# such as text, images, or audio, in a high-dimensional space.
# In the context of natural language processing (NLP),
# embeddings are often used to represent words, sentences, or documents
# in a way that captures their semantic meaning.

# An embedding turns text into a list of numbers (a vector)
# that captures the meaning of the text.

# similar meanings -> nearby vectors

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.environ["OPENROUTER_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
)
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]

def embed(text: str) -> list[float]:
    """Return the embedding vector for a piece of text."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

vec = embed("home loan eligibility")
print("text       : 'home loan eligibility'")
print("vector dim :", len(vec))
print("first 5 dim:", [round(x, 4) for x in vec[:5]])

resp = client.embeddings.create(
    model=EMBED_MODEL,
    input=["saving account interest", "how to block a stolen card"])

print("\n batched embeddings:", len(resp.data), "vectors, each dim", len(resp.data[0].embedding))
