from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
# To see the raw template string
print(prompt_template.template)
print(prompt_template.input_variables)
#assing a topic and show the prompt
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

from langchain_core.prompts import ChatPromptTemplate

# Define the chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Tell me a joke about {topic}")
])

# Fill in the template with input
formatted_chat = chat_template.invoke({"topic": "cats"})

# To see the messages that would be sent to the LLM
print(formatted_chat.to_messages())