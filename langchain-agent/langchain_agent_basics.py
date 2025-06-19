import getpass
import os
from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
model = init_chat_model("gpt-4o-mini", model_provider="openai")

#Example 1
# messages = [
#     SystemMessage("Translate the following from English into German"),
#     HumanMessage("Hello, how are you doing ?"),
# ]
#
#
# messages = [SystemMessage(content="Translate everything the user says from English into German.")]
#
# while True:
#     user_input = input("You (English): ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
#     messages.append(HumanMessage(content=user_input))
#     response = model.invoke(messages)
#     print("AI (German):", response.content)

# Example 2
print("With AI Message in history:")
messages = [HumanMessage(content="What is 12 times 4?")]
response1 = model.invoke(messages)
print("Step 1 - Answer:", response1.content)

messages.append(AIMessage(content=response1.content))

messages.append(HumanMessage(content="Now subtract 10 from that."))
response2 = model.invoke(messages)
print("Step 2 - Answer:", response2.content)

#example 3
print("Without AI Message in history:")

messages = [HumanMessage(content="What is 12 times 4?")]
response1 = model.invoke(messages)
print("Step 1 - Answer:", response1.content)

messages.append(HumanMessage(content="Now subtract 10 from that."))
response2 = model.invoke(messages)
print("Step 2 - Answer:", response2.content)