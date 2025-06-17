from dotenv import load_dotenv
import os

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from dataclasses import dataclass

load_dotenv()  # load .env into os.environ

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set or is empty.")
else:
    print("✅ API key is set.")

llm = OpenAI(model='gpt-4.1-nano',api_key=api_key)

# To retrieve a complete response
# response = llm.complete("William Shakespeare is ")
# print("Full Response:\n", response)
#

# # To stream the response token by token
# print("\nStreaming Response:")
# stream = llm.stream_complete("William Shakespeare is ")
# for token in stream:
#     print(token.delta, end="", flush=True)

#Chat interface
# messages = [
#     ChatMessage(role="system", content="You are a helpful assistant."),
#     ChatMessage(role="user", content="Tell me a joke."),
# ]
# chat_response = llm.stream_chat(messages)
# for token in chat_response:
#     print(token.delta, end="", flush=True)

#tool calling
def generate_song(name: str, artist: str,lyrics:str) -> dict:
    return {
        "name": name,
        "artist": artist,
        "lyrics": lyrics,

    }

# Wrap tool
tool = FunctionTool.from_defaults(fn=generate_song)

# Init LLM (GPT-4o is a chat model)
llm = OpenAI(model="gpt-4o")

# Run with tool
response = llm.predict_and_call(
    tools=[tool],
    user_msg="Pick a random song for me by Taylor Swift and show the song name, artist and two lines of lyrics.Use tool if required.",
)

print(response)