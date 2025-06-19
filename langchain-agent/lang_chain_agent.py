from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI


#hide warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun
# Tool 1: Built-in DuckDuckGo Web Search
search_tool = DuckDuckGoSearchRun()

# Tool 2: Simple Calculator
def calculator(expression: str) -> str:
    """Evaluate a math expression like '12 * 8 + 4'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Register tools for the agent
tools = [
    Tool(
        name="web_search",
        func=search_tool.run,
        description="Use for looking up current events, news, facts, or real-time data"
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="Use for solving math expressions like addition, multiplication, etc."
    )
]
# Define the LLM api
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

#  Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors = True
)

while True:
    query = input("\nAsk something (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    result = agent.invoke({"input": query})
    print("\n Final Answer:", result["output"])