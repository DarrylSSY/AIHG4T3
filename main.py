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

# In-memory storage for language preferences based on chat ID
language_preferences = {}

# Update the conversation history with a limit
def update_conversation_history(chat_id: str, user_query: str, bot_response: str):
    history = conversation_history.get(str(chat_id), [])
    history.append({"user": user_query, "bot": bot_response})

    # Keep the history within the defined limit
    if len(history) > CONVERSATION_LIMIT:
        history = history[-CONVERSATION_LIMIT:]

    conversation_history[str(chat_id)] = history


# Define the AI's role as a system message
SYSTEM_MESSAGE = """
You are a helpful assistant that specializes in assisting migrant workers with navigating and using the DBS Digibank app.
Your goal is to provide clear, helpful, and concise instructions about the app, including features, common issues, and how to use it effectively.
Be empathetic, patient, and focused on helping migrant workers.
Adjust accordingly to the user needs, language proficiency, and cultural background. You should be multicultural and multilingual.
Always respond in the user's preferred language, which is provided to you. If the user changes the language preference, adjust your responses accordingly.
If you suspect that the user might be facing a scam or fraud, provide clear instructions on how to verify the authenticity of the message or call.
"""

# Modify the generate_response function to include conversation limit logic and language preference
async def generate_response(user_query: str, chat_id: str):
    try:
        # Retrieve the last 3 exchanges to use as context (or fewer if there are not that many)
        history = conversation_history.get(str(chat_id), [])[-3:]

        # Get the user's language preference
        user_language = language_preferences.get(str(chat_id), 'English')

        # Modify the prompt with the conversation history, user query, and language preference
        prompt = SYSTEM_MESSAGE + f"\n\nThe user's preferred language is {user_language}.\n\n"
        for turn in history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n---\n"
        prompt += f"User: {user_query}\n"

        # You may also add a direct instruction to the model
        prompt += f"\nPlease respond in {user_language}."

        # Query the QA chain with the correct input key 'query'
        response = qa_chain.invoke({"query": prompt})
        response_text = str(response["result"]).strip()


        # Define follow-up options based on keywords in the response (translated to user's language)
        follow_up_options = []
        # Account-related options
        if "bank account" in response_text.lower() or "account opening" in response_text.lower():
            follow_up_options = [
                "How to open a bank account?",
                "Requirements for opening an account",
                "How to close a bank account?"
            ]

        # Transfer-related options
        elif "transfer money" in response_text.lower() or "payment" in response_text.lower():
            follow_up_options = [
                "How to transfer money?",
                "Fees for transferring money",
                "Transfer limits and time"
            ]

        # Balance-related options
        elif "balance" in response_text.lower():
            follow_up_options = [
                "How to check my balance?",
                "How to set balance alerts",
                "How to check previous transactions?"
            ]

        # Loan-related options
        elif "loan" in response_text.lower() or "borrow money" in response_text.lower():
            follow_up_options = [
                "How to apply for a loan?",
                "Loan interest rates",
                "What documents are needed for a loan?"
            ]

        # Card-related options
        elif "credit card" in response_text.lower() or "debit card" in response_text.lower():
            follow_up_options = [
                "How to apply for a credit card?",
                "How to block or replace a lost card?",
                "How to view credit card statements?"
            ]

        # Digibank app-related options
        elif "mobile app" in response_text.lower() or "digibank" in response_text.lower():
            follow_up_options = [
                "How to log in to DBS Digibank?",
                "How to reset my password?",
                "Features of DBS Digibank app"
            ]

        # Transaction-related options
        elif "transaction" in response_text.lower():
            follow_up_options = [
                "How to view past transactions?",
                "Dispute a transaction",
                "How to download account statements?"
            ]

        # Investment-related options
        elif "investment" in response_text.lower() or "stock" in response_text.lower() or "mutual fund" in response_text.lower():
            follow_up_options = [
                "How to invest in stocks?",
                "Investment options in DBS",
                "How to track my investments?"
            ]

        # Default option if no specific keywords are found
        else:
            follow_up_options = ["Contact support"]

        # Clean up the response text: remove any "Bot:" prefix if present
        response_text = response_text.replace("Bot:", "").strip()

        # Update the conversation history with the new interaction
        update_conversation_history(str(chat_id), user_query, response_text)

        return response_text, follow_up_options  # Return both the response and follow-up options
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return "Sorry, I am unable to respond right now.", []


# Function to create reply keyboard based on AI's follow-up options
def create_reply_keyboard(follow_up_options, user_language='English'):
    """Create a reply keyboard with the follow-up options provided by the AI."""
    # Here we can translate follow_up_options to user's language if needed
    keyboard = [[option] for option in follow_up_options]  # Create buttons in a list format
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)


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
        # Convert chat_id to string for consistent dictionary keys
        chat_id_str = str(chat_id)

        # Handle /clear command
        if user_query == "/clear":
            conversation_history.pop(chat_id_str, None)  # Clear the chat history
            language_preferences.pop(chat_id_str, None)  # Clear the language preference
            response_text = escape_markdown("Your chat history has been cleared.")
            await send_telegram_message(chat_id, response_text)
            return {"status": "ok"}

        # Handle /start command
        if user_query == "/start":
            # Provide language options
            languages = ["English", "中文", "Bahasa Indonesia", "বাংলা", "தமிழ்", "မြန်မာဘာသာ"]
            reply_markup = ReplyKeyboardMarkup([[lang] for lang in languages], one_time_keyboard=True, resize_keyboard=True)
            welcome_message = "Please select your preferred language:\n请选择您的首选语言：\nSilakan pilih bahasa pilihan Anda:\nআপনার পছন্দের ভাষা নির্বাচন করুন:\nதயவுசெய்து உங்கள் விருப்பமான மொழியைத் தேர்ந்தெடுக்கவும்:\nကျေးဇူးပြု၍ သင်နှစ်သက်သောဘာသာစကားကိုရွေးချယ်ပါ။"
            response_text = escape_markdown(welcome_message)
            await send_telegram_message(chat_id, response_text, reply_markup)
            return {"status": "ok"}

        # If the user hasn't selected a language yet
        if str(chat_id) not in language_preferences:
            # Assume the user's message is their language choice
            selected_language = user_query
            # Validate the selected language
            valid_languages = ["English", "中文", "Bahasa Indonesia", "বাংলা", "தமிழ்", "မြန်မာဘာသာ"]
            if selected_language in valid_languages:
                language_preferences[str(chat_id)] = selected_language
                # Send a confirmation message in the selected language
                confirmation_messages = {
                    "English": "Language set to English. How can I assist you today?",
                    "中文": "语言已设置为中文。请问我能为您做些什么？",
                    "Bahasa Indonesia": "Bahasa diatur ke Bahasa Indonesia. Bagaimana saya dapat membantu Anda hari ini?",
                    "বাংলা": "ভাষা বাংলা সেট করা হয়েছে। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
                    "தமிழ்": "மொழி தமிழ் அமைக்கப்பட்டது. இன்று நான் உங்களுக்கு எப்படி உதவலாம்?",
                    "မြန်မာဘာသာ": "ဘာသာစကားကို မြန်မာဘာသာ အဖြစ် သတ်မှတ်ပြီးပါပြီ။ ဒီနေ့ ကျွန်ုပ်ဘယ်လိုကူညီရမလဲ။"
                }
                response_text = escape_markdown(confirmation_messages[selected_language])
                await send_telegram_message(chat_id, response_text)
                return {"status": "ok"}
            else:
                # Prompt the user again to select a valid language
                languages = ["English", "中文", "Bahasa Indonesia", "বাংলা", "தமிழ்", "မြန်မာဘာသာ"]
                reply_markup = ReplyKeyboardMarkup([[lang] for lang in languages], one_time_keyboard=True, resize_keyboard=True)
                error_message = "Invalid selection. Please choose a language from the options below."
                response_text = escape_markdown(error_message)
                await send_telegram_message(chat_id, response_text, reply_markup)
                return {"status": "ok"}

        # Handle /contact command
        if user_query == "/contact":
            # Get user's language preference
            user_language = language_preferences.get(str(chat_id), 'English')
            contact_info = {
                "English": (
                    "For further assistance, you can contact DBS Customer Support:\n"
                    "- Phone: 1800-111-1111 (24/7)\n"
                    "- Email: support@dbs.com\n"
                    "- Visit: https://www.dbs.com/contact-us"
                ),
                "中文": (
                    "如需进一步帮助，您可以联系星展银行客户支持：\n"
                    "- 电话：1800-111-1111（全天候）\n"
                    "- 电子邮件：support@dbs.com\n"
                    "- 访问：https://www.dbs.com/contact-us"
                ),
                "Bahasa Indonesia": (
                    "Untuk bantuan lebih lanjut, Anda dapat menghubungi Dukungan Pelanggan DBS:\n"
                    "- Telepon: 1800-111-1111 (24/7)\n"
                    "- Email: support@dbs.com\n"
                    "- Kunjungi: https://www.dbs.com/contact-us"
                ),
                "বাংলা": (
                    "অতিরিক্ত সহায়তার জন্য, আপনি ডিবিএস গ্রাহক সমর্থনের সাথে যোগাযোগ করতে পারেন:\n"
                    "- ফোন: ১৮০০-১১১-১১১১ (২৪/৭)\n"
                    "- ইমেইল: support@dbs.com\n"
                    "- ভিজিট করুন: https://www.dbs.com/contact-us"
                ),
                "தமிழ்": (
                    "கூடுதல் உதவிக்கு, நீங்கள் டிபிஎஸ் வாடிக்கையாளர் ஆதரவை தொடர்பு கொள்ளலாம்:\n"
                    "- தொலைபேசி: 1800-111-1111 (24/7)\n"
                    "- மின்னஞ்சல்: support@dbs.com\n"
                    "- பார்வையிடவும்: https://www.dbs.com/contact-us"
                ),
                "မြန်မာဘာသာ": (
                    "ထပ်မံကူညီရန်အတွက် သင်သည် DBS Customer Support ကို ဆက်သွယ်နိုင်သည်။\n"
                    "- ဖုန်းနံပါတ် - ၁၈၀၀-၁၁၁-၁၁၁၁ (၂၄/၇)\n"
                    "- အီးမေးလ် - support@dbs.com\n"
                    "- ဝဘ်ဆိုက် - https://www.dbs.com/contact-us"
                )
            }
            response_text = escape_markdown(contact_info[user_language])
            await send_telegram_message(chat_id, response_text)
            return {"status": "ok"}

        # Handle user queries as normal
        response_text, follow_up_options = await generate_response(user_query, str(chat_id))

        # Get user's language preference
        user_language = language_preferences.get(str(chat_id), 'English')

        # Escape special characters for Markdown formatting
        response_text = escape_markdown(response_text)

        # Use reply keyboard for follow-up options (Note: For simplicity, follow-up options are in English)
        if follow_up_options:
            reply_markup = create_reply_keyboard(follow_up_options, user_language)
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
