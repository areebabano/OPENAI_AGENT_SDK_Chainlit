# ------------------------------------- CHAINLIT ----------------------------------------

# Required imports
import os  # To access environment variables
import asyncio  # To handle asynchronous operations

import nest_asyncio  # Allows nested async loops (useful in environments like Jupyter)
from dotenv import load_dotenv  # Loads environment variables from a .env file
import chainlit as cl  # Chainlit is used for building chat interfaces

# Importing necessary classes and functions from agents module
from agents import (
    Agent, 
    AsyncOpenAI, 
    OpenAIChatCompletionsModel, 
    RunConfig, 
    Runner
)

# Enable nested event loops (important for async environments like notebooks)
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

# If API key is not found, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI API KEY is not defined. Please check your .env file.")

# Set up the external OpenAI-compatible client using Gemini's base URL
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the model that will be used (Gemini Flash model)
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Configuration for how the agent should run
config: RunConfig = RunConfig(
    model=model,  # Model to use
    model_provider=external_client,  # Provider/client for the model
    tracing_disabled=True  # Disable tracing for this session
)

# Creating the Math Tutor agent with clear instructions
agent: Agent = Agent(
    name="Math Tutor",
    instructions="""
You are Math Tutor, an expert agent dedicated only to mathematics. 
Assist users with topics like algebra, calculus, geometry, and arithmetic. 
Do not respond to any non-math questions. 
If asked about another subject, politely decline and guide the user back to math-related queries.
"""
)

# When the chat starts
@cl.on_chat_start
async def handle_start():
    # Initialize empty conversation history in user session
    cl.user_session.set("history", [])
    # Send welcome message to user
    await cl.Message(content="Hello!").send()

# When a message is received from the user
@cl.on_message
async def handle_message(message: cl.Message):
    # Retrieve conversation history from user session
    history = cl.user_session.get("history")

    # Append the new user message to the history
    history.append({"role": "user", "content": message.content})

    # Pass the entire conversation history to the agent to get a response
    result = await Runner.run(
        agent,       # The Math Tutor agent
        input=history,  # Full conversation history
        run_config=config  # Configuration for running the agent
    )

    # Add the agent's response to the conversation history
    history.append({"role": "assistant", "content": result.final_output})
    # Save updated history back to session
    cl.user_session.set("history", history)
    # Send the agent's message as a reply
    await cl.Message(content=result.final_output).send()


# ----------------------------------- chainlit ----------------------------------------------


# # Required imports
# import os  # To access environment variables
# import asyncio  # To handle asynchronous operations

# import nest_asyncio  # Allows nested async loops (useful in environments like Jupyter)
# from dotenv import load_dotenv  # Loads environment variables from a .env file
# import chainlit as cl  # Chainlit is used for building chat interfaces

# # Importing necessary classes and functions from agents module
# from agents import (
#     Agent, 
#     AsyncOpenAI, 
#     OpenAIChatCompletionsModel, 
#     RunConfig, 
#     Runner
# )

# # Enable nested event loops (important for async environments like notebooks)
# nest_asyncio.apply()

# # Load environment variables from .env file
# load_dotenv()

# # Get the Gemini API key from environment variables
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# # If API key is not found, raise an error
# if not gemini_api_key:
#     raise ValueError("GEMINI API KEY is not defined. Please check your .env file.")

# # Set up the external OpenAI-compatible client using Gemini's base URL
# external_client: AsyncOpenAI = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# # Define the model that will be used (Gemini Flash model)
# model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# # Configuration for how the agent should run
# config: RunConfig = RunConfig(
#     model=model,  # Model to use
#     model_provider=external_client,  # Provider/client for the model
#     tracing_disabled=True  # Disable tracing for this session
# )

# # Creating the Math Tutor agent with clear instructions
# agent: Agent = Agent(
#     name="Math Tutor",
#     instructions="""
# You are Math Tutor, an expert agent dedicated only to mathematics. 
# Assist users with topics like algebra, calculus, geometry, and arithmetic. 
# Do not respond to any non-math questions. 
# If asked about another subject, politely decline and guide the user back to math-related queries.
# """
# )

