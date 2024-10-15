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

# Embeddings API
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# In-memory storage for conversation history with a limit
CONVERSATION_LIMIT = 10  # Set the history limit to 10 exchanges

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
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# In-memory storage for conversation history based on chat ID
conversation_history = {}


# Update the conversation history with a limit
def update_conversation_history(chat_id: str, user_query: str, bot_response: str):
    history = conversation_history.get(chat_id, [])
    history.append({"user": user_query, "bot": bot_response})

    # Keep the history within the defined limit
    if len(history) > CONVERSATION_LIMIT:
        history = history[-CONVERSATION_LIMIT:]

    conversation_history[chat_id] = history


# Define the AI's role as a system message
SYSTEM_MESSAGE = """
You are a helpful assistant that specializes in assisting migrant workers with navigating and using the DBS Digibank app.
Your goal is to provide clear, helpful, and concise instructions about the app, including features, common issues, and how to use it effectively.
Be empathetic, patient, and focused on helping migrant workers.
Adjust accordingly to the user needs, language proficiency, and cultural background. You should be multicultural and multilingual.
"""


# Modify the generate_response function to include conversation limit logic
async def generate_response(user_query: str, chat_id: str):
    try:
        # Retrieve the last 3 exchanges to use as context (or fewer if there are not that many)
        history = conversation_history.get(chat_id, [])[-3:]

        # Modify the prompt with the conversation history and user query
        prompt = SYSTEM_MESSAGE + "\n\n"
        for turn in history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n---\n"
        prompt += f"User: {user_query}\n"

        # Query the QA chain with the correct input key 'query'
        response = qa_chain.invoke({"query": prompt})  # Use 'query' instead of 'input'
        response_text = str(response["result"])  # Ensure the response is a string

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

        # Default option if no specific keywords are found
        else:
            follow_up_options = ["Contact support"]

        # Clean up the response text: remove any "Bot:" prefix if present
        response_text = response_text.replace("Bot:", "").strip()

        # Update the conversation history with the new interaction
        update_conversation_history(chat_id, user_query, response_text)

        return response_text, follow_up_options  # Return both the response and follow-up options
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return "Sorry, I am unable to respond right now.", []


# Function to create reply keyboard based on AI's follow-up options
def create_reply_keyboard(follow_up_options):
    """Create a reply keyboard with the follow-up options provided by the AI."""
    keyboard = [[option] for option in follow_up_options]  # Create buttons in a list format
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)


# Send a message back to the user via Telegram API with optional reply_markup
# Send a message back to the user via Telegram API with optional reply_markup
async def send_telegram_message(chat_id: str, text: str, reply_markup=None):
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Build the payload for the message
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",  # Ensure Telegram knows the message is in MarkdownV2 format
    }

    # Add reply_markup if provided
    if reply_markup:
        payload["reply_markup"] = reply_markup.to_dict()

    # Log the payload for debugging
    logging.info(f"Sending message to chat_id: {chat_id}, payload: {payload}")

    # Send the message using Telegram API
    async with httpx.AsyncClient() as client:
        response = await client.post(telegram_url, json=payload)

        # Log the response for debugging
        logging.info(f"Telegram response: {response.status_code}, {response.json()}")

        # Check if there was an error in the response
        if response.status_code != 200:
            logging.error(f"Failed to send message: {response.status_code}, {response.text}")


# Webhook endpoint for Telegram to send updates
@app.post("/webhook/")
async def telegram_webhook(request: Request):
    data = await request.json()

    # Extract the message and chat ID
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    user_query = message.get("text", "").strip()

    if chat_id and user_query:
        # Handle /clear command
        if user_query == "/clear":
            conversation_history.pop(str(chat_id), None)  # Clear the chat history
            response_text = escape_markdown("Your chat history has been cleared.")
            await send_telegram_message(chat_id, response_text)
            return {"status": "ok"}

        # Handle /start command
        if user_query == "/start":
            welcome_message = (
                "Welcome to the DBS Digibank Helper Bot! I'm here to assist you with navigating the DBS Digibank app. "
                "You can ask questions about account management, transfers, balance checks, loans, and more. "
                "Type /contact for support information."
            )
            response_text = escape_markdown(welcome_message)
            await send_telegram_message(chat_id, response_text)
            return {"status": "ok"}

        # Handle /contact command
        if user_query == "/contact":
            contact_info = (
                "For further assistance, you can contact DBS Customer Support:\n"
                "- Phone: 1800-111-1111 (24/7)\n"
                "- Email: support@dbs.com\n"
                "- Visit: https://www.dbs.com/contact-us"
            )
            response_text = escape_markdown(contact_info)
            await send_telegram_message(chat_id, response_text)
            return {"status": "ok"}

        # Handle user queries as normal
        response_text, follow_up_options = await generate_response(user_query, str(chat_id))

        # Escape special characters for Markdown formatting
        response_text = escape_markdown(response_text)

        # Use reply keyboard for follow-up options
        if follow_up_options:
            reply_markup = create_reply_keyboard(follow_up_options)
        else:
            reply_markup = None

        # Send the response back to the user
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
