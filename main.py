import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage
import httpx

# Initialize FastAPI app
app = FastAPI()

# Get the API keys from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_url = f"https://api.telegram.org/bot{telegram_token}"

# Initialize the GPT-4 chat model using LangChain's ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

# Define a function to get session history
def get_session_history():
    return []

# Initialize conversation memory with required arguments
memory = RunnableWithMessageHistory(runnable=llm, get_session_history=get_session_history)

# Data model for handling the Telegram Webhook payload
class TelegramWebhook(BaseModel):
    update_id: int
    message: dict

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to your Telegram GPT-4 Bot with FastAPI"}

# Basic endpoint for greeting
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# Function to handle the conversation with memory
async def run_conversation(user_input: str):
    # Get the previous conversation history from memory
    conversation_history = memory.get_session_history

    # Generate a response from GPT-4 based on the input and past conversation
    response = await llm.invoke([HumanMessage(user_input)], previous_messages=conversation_history)

    # Update memory with the new conversation
    memory.add_message(HumanMessage(user_input))
    memory.add_message(AIMessage(response))

    return response

# Endpoint for receiving Telegram messages via webhook
@app.post("/webhook/")
async def telegram_webhook(webhook: TelegramWebhook):
    message = webhook.message.get("text", "")
    chat_id = webhook.message["chat"]["id"]

    # Run the conversation handler to get GPT-4's response
    response_text = await run_conversation(message)

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