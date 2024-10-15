import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import httpx

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_url = f"https://api.telegram.org/bot{telegram_token}"

# Define the chatbot function to generate a response
async def generate_response(user_query: str):
    try:
        # Use the query to generate a response using GPT-4
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query}
            ]
        )

        # Return the GPT-4 generated answer using model_dump
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
