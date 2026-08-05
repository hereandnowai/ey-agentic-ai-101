import gradio as gr

from chatbot import (
    DOCUMENT_PATH,
    DOCUMENT_URL,
    MODEL,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    get_streaming_response,
)

def chat_interface(message, history):
    thinkng_text = ""
    response_text = ""

    for kind, text in get_streaming_response(message):
        if kind == "thinking":
            thinkng_text += text
            yield f"<details open><summary> Thinking...</summary>\n\n{thinkng_text}\n\n</details>"
        elif kind == "response":
            response_text += text
            yield (
                f"<details><summary> Thinking (done)</summary>\n\n{thinkng_text}\n\n</details>"
                f"\n\n{response_text}"
            )

demo = gr.ChatInterface(
    fn=chat_interface,
    chatbot=gr.Chatbot(
        height=500,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "\\(", "right": "\\)", "display": False},
            {"left": "$", "right": "$", "display": False},
        ],
    ),
    title=f"Chat with Caramel AI {DOCUMENT_PATH.name} using {MODEL}",
    description=(
        f"Ask questions about the document {DOCUMENT_PATH.name} (or {DOCUMENT_URL}) and get answers from the LLM {MODEL} "
    ),
    examples=[
        ["What is KDA?"],
        ["How does Kimi Linear compare to full attention?"],
        ["What is the 3:1 ratio in paper mentioned?"],
        ["How much does it reduce the KV cache?"]
    ]
)

if __name__ == "__main__":
    demo.launch()