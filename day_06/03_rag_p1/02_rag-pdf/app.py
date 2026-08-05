import gradio as gr
from chatbot import get_streaming_response, PDF_PATH

def chat_interface(message, history):
    response_text = ""

    for kind, text in get_streaming_response(message):
        if kind == "response":
            response_text += text
            yield response_text

demo = gr.ChatInterface(
    fn=chat_interface,
    title="Caramel AI - PDF Document RAG BOT",
    description=f"Ask questions about the content of the loaded PDF doucment: {PDF_PATH}"
)

if __name__ == "__main__":
    demo.launch()