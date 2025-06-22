from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableMap

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#hide warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

tweet_prompt = PromptTemplate.from_template("Write a short tweet about {topic}")
tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt)

translate_prompt = PromptTemplate.from_template("Translate this to German:\n\n{tweet}")
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

# 4. Define a parallel chain that runs two operations at once
parallel = RunnableMap({
    "tweet": tweet_chain,
}) | RunnableMap({
    "tweet": lambda x: x["tweet"],
    "german": translate_chain.with_config(run_name="TranslateChain")
})

result = parallel.invoke({"topic": "LangChain is awesome"})

print("\nTweet:", result["tweet"])
print("German Translation:", result["german"])