from os import getenv  # Brings in a tool for reading settings from the computer.
from pathlib import Path  # Brings in a tool for working with file locations.
import requests  # Brings in a tool for downloading files from the internet.
from dotenv import load_dotenv  # Brings in a tool for loading private settings from a file.
from langchain_chroma import Chroma  # Brings in the tool that stores and searches text pieces.
from langchain_community.document_loaders import PyPDFLoader  # Brings in a tool for reading PDF files.
from langchain_core.output_parsers import StrOutputParser  # Brings in a tool for turning answers into plain text.
from langchain_core.prompts import ChatPromptTemplate  # Brings in a tool for building chat messages.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Brings in tools for chatting and making text numbers.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Brings in a tool for cutting long text into pieces.
from pydantic import SecretStr  # Brings in a tool for keeping the API key protected.
from system_prompt import SYSTEM_PROMPT  # Brings in the instructions given to the chatbot.

load_dotenv()  # Loads the settings stored in the local environment file.

OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")  # Reads the secret key used to contact OpenRouter.
OPENROUTER_BASE_URL = getenv("OPENROUTER_BASE_URL")  # Reads the web address used to contact OpenRouter.
MODEL = getenv("MODEL", "openai/gpt-5.6-luna")  # Reads the chat model name, or uses a default name.
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")  # Reads the text-number model name, or uses a default.

DOCUMENT_URL = "https://arxiv.org/pdf/2510.26692"  # Stores the web address of the paper to download.
DOCUMENT_PATH = Path(__file__).parent / "kimi.pdf"  # Builds the location where the paper will be saved.
CHROMA_PATH = Path(__file__).parent / "chroma_db"  # Builds the location where the search data will be saved.
COLLECTION_NAME = f"kimi_{MODEL.replace('/', '_')}_{EMBEDDING_MODEL.replace('/', '_')}"  # Creates a unique name for this set of search data.

if not OPENROUTER_API_KEY:  # Checks whether the secret key is missing.
    raise RuntimeError("OPENROUTER_API_KEY not found. Add it to your .env file.")  # Stops the program and explains how to fix the missing key.

API_KEY = SecretStr(OPENROUTER_API_KEY)  # Wraps the key so it is less likely to be shown by accident.

# step 1: download the PDF document
def download_pdf(url, file_path):  # Defines a reusable action for downloading the paper.
    if file_path.exists():  # Checks whether the paper was already downloaded.
        print(f"Using cached PDF: {file_path.name}")  # Tells the user that the saved copy will be used.
        return  # Leaves the action because there is no need to download the paper again.

    print(f"Downloading {url} ...")  # Tells the user that the download has started.
    response = requests.get(url, timeout=60)  # Downloads the paper and waits up to one minute.
    response.raise_for_status()  # Stops with an error if the download did not work.
    file_path.write_bytes(response.content)  # Saves the downloaded paper on the computer.
    print(f"Saved {file_path.name} ({len(response.content) / 1_000_000:.1f} MB)")  # Tells the user that the paper was saved.

download_pdf(DOCUMENT_URL, DOCUMENT_PATH)  # Downloads the paper unless a saved copy already exists.

# step 2 - llm setup & embedding setup
llm = ChatOpenAI(  # Creates the chat model connection using the chosen settings.
    model=MODEL,
    api_key=API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0,
    default_headers={"X-Title": "Document RAG on Kimi Paper"}
)

emdeddings = OpenAIEmbeddings(  # Creates the tool that changes text into searchable numbers.
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
)

# step 3 - load the PDF document and split it into chunks as vectors
def build_vector_store():  # Defines an action that loads or builds the paper search data.
    store = Chroma(  # Opens the saved search data or prepares a new collection.
        collection_name=COLLECTION_NAME,
        embedding_function=emdeddings,
        persist_directory=str(CHROMA_PATH)
    )
    if store.get(limit=1)["ids"]:  # Checks whether this collection already contains saved text pieces.
        print(f"Loaded saved vector store from {CHROMA_PATH.name}/")  # Tells the user that saved search data was found.
        return store  # Gives back the saved search data without rebuilding it.

    print(f"Indexing: {DOCUMENT_PATH.name} ...")  # Tells the user that the paper is being prepared for searching.
    docs = PyPDFLoader(str(DOCUMENT_PATH)).load()  # Reads the paper and turns its pages into text.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Makes a tool for cutting the text into overlapping pieces.
    chunks = splitter.split_documents(docs)  # Cuts the paper into smaller pieces for searching.
    store.add_documents(chunks)  # Saves the text pieces as searchable numbers.
    print(f"Indexed {len(chunks)} chunks with {EMBEDDING_MODEL} into {CHROMA_PATH.name}/")  # Tells the user how many pieces were saved.
    return store  # Gives back the newly built search data.

vector_store = build_vector_store()  # Loads or builds the searchable paper data.
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Makes a searcher that returns the five best matching pieces.

# step 4 - create a RAG chain
prompt = ChatPromptTemplate.from_messages(  # Builds the message format used for each question.
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

def format_doc(docs):  # Defines an action for joining search results into one readable block.
    return "\n\n---\n\n".join(doc.page_content for doc in docs)  # Joins each piece of text with a visible separator.

rag_chain = prompt | llm | StrOutputParser()  # Connects the prompt, chat model, and plain-text answer parser.

# step 5 - stream the answer
def get_streaming_response(user_input):  # Defines an action that finds information and sends back the answer piece by piece.
    docs = retriever.invoke(user_input)  # Searches the paper for pieces related to the user's question.
    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in docs})  # Collects and sorts the page numbers that were found.
    yield ("thinking", f"Found {len(docs)} relevant chunks (pages: {pages}).\n")  # Reports how many matching pieces were found.
    yield ("thinking", f"Asking {MODEL} via OpenRouter ...\n")  # Reports which chat model will write the answer.

    for text in rag_chain.stream({"context": format_doc(docs), "question": user_input}):  # Sends the question and paper context to the model.
        if text:  # Checks whether the model sent back some text.
            yield ("response", text)  # Passes the answer piece to the person using the program.

# step 6 - run it in the terminal
if __name__ == "__main__":  # Runs the chat loop only when this file is started directly.
    print(f"\nChat with Caramel AI {DOCUMENT_PATH.name} using {MODEL}. Type 'quit' to exit.\n")  # Welcomes the user and explains how to leave.
    while True:  # Keeps asking questions until the user chooses to stop.
        user_input = input("You: ")  # Reads the next question typed by the user.
        if user_input.lower() in {"quit", "exit"}:  # Checks whether the user asked to leave.
            break  # Ends the chat loop.
        for kind, text in get_streaming_response(user_input):  # Receives each progress message and answer piece.
            print(text, end="", flush=True)  # Shows each piece immediately without waiting for the whole answer.
        print("\n")  # Adds space before the next question.