from dotenv import load_dotenv
import os
import asyncio
import json
from openai import AsyncOpenAI

async def run_agent():
    # Load .env
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please set it in .env")

    # Create Gemini client
    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

    # Define tool function
    def add_numbers(a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b

    # Tool schema for OpenAI
    tools = [{
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }]

    # Initial chat completion
    response = await client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can solve math by calling tools."},
            {"role": "user", "content": "What is 42 plus 58?"}
        ],
        tools=tools,
        tool_choice="auto"
    )

    # Handle tool calls
    message = response.choices[0].message
    
    if message.tool_calls:
        # Execute the tool call
        tool_call = message.tool_calls[0]
        if tool_call.function.name == "add_numbers":
            args = json.loads(tool_call.function.arguments)
            result = add_numbers(args["a"], args["b"])
            
            # Send tool result back to get final response
            final_response = await client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can solve math by calling tools."},
                    {"role": "user", "content": "What is 42 plus 58?"},
                    {"role": "assistant", "content": message.content, "tool_calls": message.tool_calls},
                    {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
                ]
            )
            print(f"\nFinal Output: {final_response.choices[0].message.content}")
        else:
            print(f"\nFinal Output: Unknown tool call")
    else:
        print(f"\nFinal Output: {message.content}")

def run():
    asyncio.run(run_agent())

if __name__ == "__main__":
    run()