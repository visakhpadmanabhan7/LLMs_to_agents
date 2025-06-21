from dotenv import load_dotenv
import warnings
load_dotenv()

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory,ConversationSummaryMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
warnings.filterwarnings("ignore")

# Define the LLM
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# Define tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events or factual queries."
    )
]

# Add memory
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

# memory= ConversationBufferWindowMemory(
#     memory_key="chat_history",
#     k=2,
#     return_messages=True
# )

memory= ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)
#  Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION ,
    memory=memory,
    # verbose=True,
    handle_parsing_errors=True
)

# Interaction loop
while True:
    user_input = input("\nAsk something (or type 'exit'): ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = agent.invoke({"input": user_input})
    print("\nFinal Answer:", response["output"])
    print("\n🧠 Memory Summary:\n", memory.buffer)