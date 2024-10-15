import os
from fastapi import FastAPI, Request
import openai
from langchain import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from pydantic import BaseModel
import httpx

# Initialize FastAPI app
app = FastAPI()

# Set up environment variables for your API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_url = f"https://api.telegram.org/bot{telegram_token}"

# Initialize conversation memory with LangChain
memory = ConversationBufferMemory()

# Set up LangChain's conversation chain using GPT-4
llm = OpenAI(model_name="gpt-4", openai_api_key=openai.api_key)
conversation = ConversationChain(llm=llm, memory=memory)

# Data model for Telegram webhook
class TelegramWebhook(BaseModel):
    update_id: int
    message: dict

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to your Telegram GPT-4 Bot with FastAPI"}

# Basic endpoint for greeting (already existing)
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# Endpoint for Telegram to send messages to (webhook URL)
@app.post("/webhook/")
async def telegram_webhook(webhook: TelegramWebhook):
    message = webhook.message.get("text", "")
    chat_id = webhook.message["chat"]["id"]

    # Generate a response using GPT-4 with LangChain memory
    response_text = conversation.run(message)

    # Send the response back to the user via Telegram
    await send_message(chat_id, response_text)

    return {"status": "ok"}

# Utility function to send a message back to Telegram user
async def send_message(chat_id: int, text: str):
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    async with httpx.AsyncClient() as client:
        await client.post(f"{telegram_url}/sendMessage", json=payload)