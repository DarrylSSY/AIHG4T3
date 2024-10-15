import os
import logging
import httpx
from fastapi import FastAPI, Request
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import re
import json

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define OpenAI API key and initialize the embeddings model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Path to your folder with PDF documents
PDF_FOLDER = "./sources"

def escape_markdown(text: str) -> str:
    """Escapes special characters in text for Markdown v2 formatting."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

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
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

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

# Modify the generate_response function to include options and use them in buttons
async def generate_response(user_query: str, chat_id: str):
    try:
        # Retrieve conversation history and limit to last 3 exchanges to avoid overwhelming the model
        history = conversation_history.get(chat_id, [])[-3:]

        # Modify the prompt to explicitly ask the AI to generate a JSON-formatted response
        prompt = SYSTEM_MESSAGE + "\n\n"
        for turn in history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n---\n"
        prompt += f"User: {user_query}\n"

        # Instruct AI to respond in JSON format
        prompt += 'Please respond in the following JSON format:\n'
        prompt += '{"response": "<main response>", "options": ["<follow-up question 1>", "<follow-up question 2>"]}\n'

        # Query the QA chain with the user's input + conversation history as context
        response = qa_chain.run(prompt)

        # Attempt to parse the response as JSON
        try:
            response_json = json.loads(response)  # Parse the AI's response as JSON
            response_text = response_json.get('response', 'Sorry, something went wrong.')
            follow_up_options = response_json.get('options', [])
        except json.JSONDecodeError:
            # If there's an issue with the JSON response, return a fallback message
            response_text = "Sorry, I couldn't understand the response format."
            follow_up_options = []

        # Clean up the response (if needed) and update the conversation history
        response_text = response_text.replace("Bot:", "").strip()
        history.append({"user": user_query, "bot": response_text})
        conversation_history[chat_id] = history

        return response_text, follow_up_options  # Return both response and follow-up options
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return "Sorry, I am unable to respond right now.", []

# Function to create inline keyboard based on AI's follow-up options
def create_inline_keyboard(follow_up_options):
    """Create an inline keyboard with the follow-up options provided by the AI."""
    keyboard = [
        [InlineKeyboardButton(option, callback_data=option) for option in follow_up_options]
    ]
    return InlineKeyboardMarkup(keyboard)

# Telegram bot token from BotFather
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Send a message back to the user via Telegram API with optional reply_markup
async def send_telegram_message(chat_id: str, text: str, reply_markup=None):
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": reply_markup.to_dict() if reply_markup else None,
        "parse_mode": "MarkdownV2"  # Ensure Telegram knows the message is in Markdown format
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
        # Generate a response using the chatbot logic (which also returns follow-up options)
        response_text, follow_up_options = await generate_response(user_query, str(chat_id))

        # Escape special characters in the response for Markdown formatting
        response_text = escape_markdown(response_text)

        # Use Inline Keyboard with follow-up options
        if len(follow_up_options) > 0:
            reply_markup = create_inline_keyboard(follow_up_options)
        else:
            reply_markup = None

        # Send the response back to the user via Telegram with Markdown parsing
        await send_telegram_message(chat_id, response_text, reply_markup)

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
