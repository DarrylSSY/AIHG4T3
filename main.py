import logging
import os
from fastapi import FastAPI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import HumanMessage, AIMessage
import httpx
from typing import Annotated
from typing_extensions import TypedDict

# Initialize FastAPI app
app = FastAPI()

# Get the API keys from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_url = f"https://api.telegram.org/bot{telegram_token}"

# Initialize the GPT-4 chat model using LangChain's ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key
)

# Initialize LangGraph MemorySaver for memory persistence
memory = MemorySaver()

# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

    def clear_messages(self):
        self["messages"] = []

# Build the state graph
graph_builder = StateGraph(State)


# Define the chatbot function
# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[])
graph_builder.add_node("tools", tool_node)

# Add edges to the graph
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph with the MemorySaver checkpointer
graph = graph_builder.compile(checkpointer=memory)

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
    try:
        # Define the system message (role assignment)
        system_message = SystemMessage(content="You are a DBS digibank chatbot guide. Your role is to assist migrant workers in using the digibank app.")

        # Retrieve conversation history from memory
        conversation_history = memory.get_history() or []

        # Add the system message and the user input to the conversation history
        if not conversation_history:
            conversation_history.append(system_message)  # Add system message only once at the beginning

        conversation_history.append(HumanMessage(user_input))

        # Pass the entire conversation history as 'messages'
        response = await llm.invoke(conversation_history)

        # Add the new user message and AI response to memory
        memory.add_message(HumanMessage(user_input))
        memory.add_message(AIMessage(response.content))  # Access the response content

        return response.content  # Return the AI's response

    except Exception as e:
        logging.error(f"Error during LLM invocation: {e}")
        return "Sorry, I am unable to respond right now."

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