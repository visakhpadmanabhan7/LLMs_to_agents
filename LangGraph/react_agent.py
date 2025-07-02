from typing import Annotated, Sequence
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4.1-nano").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + list(state.messages))
    return AgentState(messages=[response])


def should_continue(state: AgentState) -> str:
    last_message = state.messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages": [HumanMessage(content="Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]
}
app.get_graph().print_ascii()

# print_stream(app.stream(inputs, stream_mode="values"))
for update in app.stream(inputs, stream_mode="updates"):
    for node, data in update.items():
        print(f"\n Node executed: {node}")

        if "messages" in data:
            for msg in data["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(" Tool calls:")
                    for call in msg.tool_calls:
                        print(f" - Tool: {call['name']}({call['args']})")
                elif hasattr(msg, "content"):
                    print(f" Response: {msg.content}")
#save graph image
