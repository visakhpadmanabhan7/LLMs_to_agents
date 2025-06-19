import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Load OpenAI key from .env
load_dotenv()

# Define the tool
@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the first result."""
    with DDGS() as ddgs:
        results = ddgs.text(query)
        for r in results:
            return f"{r['title']}: {r['body']}"
    return "No relevant results found."

# Initialize the model and bind tools
model = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
model_with_tools = model.bind_tools([web_search])

messages = [HumanMessage(content="What's the weather like in Tokyo today?")]
response = model_with_tools.invoke(messages)
# Handle tool call
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "web_search":
        output = web_search.invoke(tool_args)
        print("🔎 Web Search Output:", output)
else:
    print("💬 Model Output:", response.content)