import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import httpx
from typing import Annotated
from typing_extensions import TypedDict

# Initialize FastAPI app
app = FastAPI()

# Get the API keys from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_url = f"https://api.telegram.org/bot{telegram_token}"

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Ensure the correct folder is set for PDFs
pdf_folder = "sources"
if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"PDF folder '{pdf_folder}' not found.")

# Load your PDFs and create a vector store
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Load and process the PDFs
documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load_and_split())

# Create embeddings for the documents and store in FAISS
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embedding_model)
vector_store.save_local("faiss_index")

# Define a prompt template for GPT-4
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="Given the context below, answer the user's question.\n\nContext: {context}\n\nQuestion: {query}"
)

# Define the chatbot function to retrieve relevant info and generate a response
async def generate_response(user_query: str):
    # Retrieve relevant chunks from the vector store
    docs = vector_store.similarity_search(user_query, k=5)
    context = " ".join([doc.page_content for doc in docs])

    # Generate the response using GPT-4
    chain = LLMChain(llm=OpenAI(model_name="gpt-4"), prompt=prompt_template)
    response = chain.run({"context": context, "query": user_query})

    return response

# Data model for handling the Telegram Webhook payload
class TelegramWebhook(BaseModel):
    update_id: int
    message: dict

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to your Telegram GPT-4 Bot with FastAPI"}

# Endpoint for receiving Telegram messages via webhook
@app.post("/webhook/")
async def telegram_webhook(webhook: TelegramWebhook):
    message = webhook.message.get("text", "")
    chat_id = webhook.message["chat"]["id"]

    # Run the conversation handler to get GPT-4's response
    response_text = await generate_response(message)

    # Send the generated response back to the user on Telegram
    await send_message(chat_id, response_text)

    return {"status": "ok"}

# Utility function to send a message back to the Telegram user
async def send_message(chat_id: int, text: str):
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    async with httpx.AsyncClient() as client:
        await client.post(f"{telegram_url}/sendMessage", json=payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
