import os
import uuid
import logging
import httpx
import requests
from fastapi import FastAPI, Request
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import for embeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define OpenAI API key and initialize the embeddings model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Path to your folder with PDF documents
PDF_FOLDER = "./sources"


# Function to load and process PDFs
def load_pdfs_and_create_vectorstore(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(loader.load())

    # Split the text into smaller chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create Chroma vector store and store the documents
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore


# Load PDFs and create vector store
vectorstore = load_pdfs_and_create_vectorstore(PDF_FOLDER)

# Initialize the language model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# In-memory storage for conversation history based on chat ID
conversation_history = {}

# Define the AI's role as a system message
SYSTEM_MESSAGE = """
You are a helpful assistant that specializes in assisting migrant workers with navigating and using the DBS Digibank app.
Your goal is to provide clear, helpful, and concise instructions about the app, including features, common issues, and how to use it effectively.
Be empathetic, patient, and focused on helping migrant workers.
"""


# Define the chatbot function to generate a response using RetrievalQA chain and conversation history
async def generate_response(user_query: str, chat_id: str):
    try:
        # Retrieve conversation history
        history = conversation_history.get(chat_id, [])

        # Combine the system message and conversation history into the prompt for the model
        prompt = SYSTEM_MESSAGE + "\n"
        for turn in history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        prompt += f"User: {user_query}\n"

        # Query the QA chain with the user's input + conversation history as context
        response = qa_chain.run(prompt)

        # Clean up the response: Remove "Bot:" from the start of the response if present
        response = response.replace("Bot:", "").strip()

        # Update conversation history
        history.append({"user": user_query, "bot": response})
        conversation_history[chat_id] = history

        return response
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return "Sorry, I am unable to respond right now."


# Telegram bot token from BotFather
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# Send a message back to the user via Telegram API
async def send_telegram_message(chat_id: str, text: str):
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    async with httpx.AsyncClient() as client:
        await client.post(telegram_url, json=payload)


# Webhook endpoint for Telegram to send updates
@app.post("/webhook/")
async def telegram_webhook(request: Request):
    data = await request.json()

    # Extract the message and chat ID from the Telegram update
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    user_query = message.get("text", "")

    if chat_id and user_query:
        # Generate a response using the chatbot logic
        response_text = await generate_response(user_query, str(chat_id))

        # Send the response back to the user via Telegram
        await send_telegram_message(chat_id, response_text)

    return {"status": "ok"}


# Health check endpoint for diagnostics
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Run the FastAPI app on the correct port for Railway
if __name__ == "__main__":
    import uvicorn

    # Railway will dynamically assign a port, default to 8000 if not found
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
