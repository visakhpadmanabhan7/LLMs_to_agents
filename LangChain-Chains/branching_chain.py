from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

#hide warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

import re


math_prompt = PromptTemplate.from_template("Calculate: {query}")
math_chain = LLMChain(llm=llm, prompt=math_prompt)

translate_prompt = PromptTemplate.from_template("Translate to German:\n{query}")
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

fallback_prompt = PromptTemplate.from_template("Cannot process mixed input: {query}")
fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)

# Router logic
def input_router(input):
    query = input["query"]

    is_number = re.fullmatch(r"[0-9+\-*/(). ]+", query)
    is_text = re.fullmatch(r"[a-zA-Z ,.'\"!?]+", query)

    if is_number:
        return math_chain
    elif is_text:
        return translate_chain
    else:
        return fallback_chain

branching_chain = RunnableLambda(input_router)

# Examples
print("\nMath:", branching_chain.invoke({"query": "12 + 8"}))
print("\nText:", branching_chain.invoke({"query": "I love LangChain"}))
print("\nMixed:", branching_chain.invoke({"query": "Translate 5 apples"}))
