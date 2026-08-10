import gradio as gr

from chatbot import ask, MODEL

def respond(message: str, history: list):
    reply, report = ask(message, history or [])
    history = (history or []) + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": reply}]
    return history, "", report

with gr.Blocks(title="Observable Chatbot") as demo:
    gr.Markdown(f"# Meridian Bank Assistant\n`{MODEL}` | one turn = one trace")

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=420, label="Chat")
            box = gr.Textbox(placeholder="What's the balance on SB-9001?",
                             show_label=False, submit_btn=True)
            gr.Examples(["What's the balance on SB-9001?",
                         "What are your Saturday timings?",
                         "And SB-9003"], inputs=box)

        with gr.Column(scale=2):
            gr.Markdown("### What that cost")
            report = gr.Markdown("_Send a message._")
            gr.Markdown("Open the trace. A balance question made **two** model calls:" \
                        "one to pick the tool, one to answer from its result." \
                        "The second prompt is bigger - it carries the tool's output.")

    box.submit(respond, [box, chat], [chat, box, report])

if __name__ == "__main__":
    demo.launch()