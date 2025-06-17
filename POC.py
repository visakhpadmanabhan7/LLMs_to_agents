from dotenv import load_dotenv
import os

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI

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
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.stream_chat(messages)
for token in chat_response:
    print(token.delta, end="", flush=True)