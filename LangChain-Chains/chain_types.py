from langchain.chains import LLMChain,SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


#hide warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
prompt1 = PromptTemplate.from_template("Write a tweet about {topic}")
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt1,output_key="tweet")

prompt2= "Translate to German"
prompt2 = PromptTemplate.from_template("Translate this to German:\n\n{tweet}")
chain2 = LLMChain(llm=llm, prompt=prompt2,output_key="german_tweet")

sequential_chain = SequentialChain(
    chains=[chain, chain2],
    input_variables=["topic"],
    output_variables=["tweet", "german_tweet"],
    verbose=True
)

result = sequential_chain.invoke({"topic": "cricket"})
print("\nOriginal Tweet:", result["tweet"])
print("\nGerman Translation:", result["german_tweet"])