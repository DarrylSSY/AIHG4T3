import logging
import os
import re
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from telegram import ReplyKeyboardMarkup

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define OpenAI API key and initialize the embeddings model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Telegram bot token from BotFather
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Path to your folder with PDF documents
PDF_FOLDER = "./sources"

def escape_markdown(text: str) -> str:
    """Escapes special characters in text for Markdown v2 formatting."""
    escape_chars = r'_[]()~`>#+-=|{}.!'
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
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

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

# Modify the generate_response function to use the correct input key 'query'
async def generate_response(user_query: str, chat_id: str):
    try:
        # Retrieve conversation history and limit to last 3 exchanges to avoid overwhelming the model
        history = conversation_history.get(chat_id, [])[-3:]

        # Modify the prompt with the conversation history and user query
        prompt = SYSTEM_MESSAGE + "\n\n"
        for turn in history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n---\n"
        prompt += f"User: {user_query}\n"

        # Query the QA chain with the correct input key 'query' (not 'input')
        response = qa_chain.invoke({"query": prompt})  # Use 'query' instead of 'input'
        response_text = str(response)  # Ensure response is a string

        # Define follow-up options based on keywords in the response
        follow_up_options = []
        # Account-related options
        if "bank account" in response_text.lower() or "account opening" in response_text.lower():
            follow_up_options = ["How to open a bank account?", "Requirements for opening an account",
                                 "How to close a bank account?"]

        # Transfer-related options
        elif "transfer money" in response_text.lower() or "payment" in response_text.lower():
            follow_up_options = ["How to transfer money?", "Fees for transferring money", "Transfer limits and time"]

        # Balance-related options
        elif "balance" in response_text.lower():
            follow_up_options = ["How to check my balance?", "How to set balance alerts",
                                 "How to check previous transactions?"]

        # Loan-related options
        elif "loan" in response_text.lower() or "borrow money" in response_text.lower():
            follow_up_options = ["How to apply for a loan?", "Loan interest rates",
                                 "What documents are needed for a loan?"]

        # Card-related options
        elif "credit card" in response_text.lower() or "debit card" in response_text.lower():
            follow_up_options = ["How to apply for a credit card?", "How to block or replace a lost card?",
                                 "How to view credit card statements?"]

        # Digibank app-related options
        elif "mobile app" in response_text.lower() or "digibank" in response_text.lower():
            follow_up_options = ["How to log in to DBS Digibank?", "How to reset my password?",
                                 "Features of DBS Digibank app"]

        # Transaction-related options
        elif "transaction" in response_text.lower():
            follow_up_options = ["How to view past transactions?", "Dispute a transaction",
                                 "How to download account statements?"]

        # Investment-related options
        elif "investment" in response_text.lower() or "stock" in response_text.lower() or "mutual fund" in response_text.lower():
            follow_up_options = ["How to invest in stocks?", "Investment options in DBS",
                                 "How to track my investments?"]

        # Default options if no specific keywords are found
        else:
            follow_up_options = ["Contact support"]

        # Clean up the response: Remove "Bot:" from the start of the response if present
        response_text = response_text.replace("Bot:", "").strip()

        # Update conversation history with the latest interaction
        history.append({"user": user_query, "bot": response_text})
        conversation_history[chat_id] = history

        return response_text, follow_up_options  # Return both response and follow-up options
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return "Sorry, I am unable to respond right now.", []


# Function to create reply keyboard based on AI's follow-up options
def create_reply_keyboard(follow_up_options):
    """Create a reply keyboard with the follow-up options provided by the AI."""
    keyboard = [[option] for option in follow_up_options]  # Create buttons in a list format
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

# Send a message back to the user via Telegram API with optional reply_markup
async def send_telegram_message(chat_id: str, text: str, reply_markup=None):
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": reply_markup.to_dict() if reply_markup else None,
        "parse_mode": "MarkdownV2"  # Ensure Telegram knows the message is in MarkdownV2 format
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

        # Use Reply Keyboard with follow-up options
        if len(follow_up_options) > 0:
            reply_markup = create_reply_keyboard(follow_up_options)
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
