from dotenv import load_dotenv
import os
from agents import Agent, Runner, set_tracing_disabled, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# Disable tracing
set_tracing_disabled(True)

# Load .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY. Please set it in .env")

# Create Gemini client (async, needed for agents package)
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
)

# Define the model wrapper
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client,
)

# Define the agent
agent = Agent(
    name="HelloAgent",
    instructions="You are a helpful assistant.",
    model=model,
)

# Run task synchronously (agents handles the asyncio event loop for us)
result = Runner.run_sync(agent, "Introduce yourself in one sentence.")

print(result.final_output)
