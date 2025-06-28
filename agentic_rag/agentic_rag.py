# agentic_rag.py

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from db_indexing import check_index_exists
from tools import load_tools
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

#  Ensure indexes exist
check_index_exists()

#  Load tools
tools = load_tools()

#  Create LLM and agent
llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
print(tools)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
agent_kwargs={
        "prefix": (
            "You are a helpful agent that decides which tool to use based on the topic. "
            "You should prefer using internal document tools (like FoodRetriever, SportsRetriever, etc.) "
            "when relevant, and only fall back to web_search if nothing fits."
        )
}
)


while True:
    query = input("\nAsk something (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    result = agent.invoke({"input": query})
    print("\nFinal Answer:", result["output"])