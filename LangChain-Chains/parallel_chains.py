from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv


#hide warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# Chain 1: Tweet generation
tweet_prompt = PromptTemplate.from_template("Write a short tweet about {topic}")
tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, output_key="tweet")

# Chain 2: Hashtag generation
hashtag_prompt = PromptTemplate.from_template("Suggest 3 hashtags for a post about {topic}")
hashtag_chain = LLMChain(llm=llm, prompt=hashtag_prompt, output_key="hashtags")

# Chain 3: Translation (note it takes tweet from chain 1)
translate_prompt = PromptTemplate.from_template("Translate this tweet into German:\n\n{tweet}")
translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="german_tweet")

input = {"topic": "football"}

# Run tweet generation first
tweet_result = tweet_chain.invoke(input)
tweet_text = tweet_result["tweet"]

parallel = RunnableMap({
    "german_tweet": translate_chain,
    "hashtags": hashtag_chain
})

#Run parallel chain
parallel_result = parallel.invoke({**input, "tweet": tweet_text})

print("Tweet:", tweet_text)
print("Hashtags:", parallel_result["hashtags"])
print("German Translation:", parallel_result["german_tweet"])