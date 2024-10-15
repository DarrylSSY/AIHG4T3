import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import httpx
from typing import Annotated
from typing_extensions import TypedDict

# Initialize FastAPI app
app = FastAPI()

# Instantiate the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_url = f"https://api.telegram.org/bot{telegram_token}"

# Define the chatbot function to retrieve relevant info and generate a response
async def generate_response(user_query: str):
    try:
        # Perform embedding-based document search (if applicable)
        search_results = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_query
        )

        # Retrieve the relevant content from the embeddings (adapt this as per your vector store logic)
        retrieved_content = " ".join([doc['text'] for doc in search_results.data])

        # Generate a response using GPT-4 based on the retrieved content and user query
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {retrieved_content}. Question: {user_query}"}
            ]
        )

        # Return the GPT-4 generated answer
        return gpt_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        return "Sorry, I am unable to respond right now."

# Data model for handling the Telegram Webhook payload
class TelegramWebhook(BaseModel):
    update_id: int
    message: dict

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to your Telegram GPT-4 Bot with OpenAI"}

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
