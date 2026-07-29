import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"]
)
MODEL = os.environ["OPENROUTER_EMBEDDING_MODEL"]

# part 1 - turn words into vectors, then into a picture
def word_map(text):
    """Take typed words, fetch their meanings, draw them on a flat map."""
    words = [w.strip() for w in text.replace(",", " ").split() if w.strip()]
    if len(words) < 3:
        raise gr.Error("Please enter at least 3 words.")

    answer = client.embeddings.create(model=MODEL, input=words)
    vectors = np.array([row.embedding for row in answer.data])

    flat = PCA(n_components=2, random_state=0).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(flat[:, 0], flat[:, 1], s=140, c='blue', zorder=3)
    for word, (x, y) in zip(words, flat):
        ax.annotate(word, (x, y), fontsize=12, xytext=(8, 5),
                    textcoords='offset points')
    ax.set_title(f"{len(words)} words, {vectors.shape[1]} numbers each")
    ax.grid(alpha=0.25)
    ax.margins(0.18)
    return fig

# part 2 - make a web app
app = gr.Interface(
    fn=word_map,
    inputs=gr.Textbox(label="Words, separated by commas", value="loan, debt, credit, guitar, violin, drum"),
    outputs=gr.Plot(label="Word Map"),
    title="Word Map - close dots mean similar meaning",
    description="Type in a list of words, separated by commas. The app will fetch their meanings and draw them on a flat map. Close dots mean similar meaning."
)

if __name__ == "__main__":
    app.launch()