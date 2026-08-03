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
MODEL = os.environ["MODEL"]
