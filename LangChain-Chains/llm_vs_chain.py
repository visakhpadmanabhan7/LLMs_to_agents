from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

#
# Plain Prompt (No Chaining)
plain_prompt = """
Summarize the blog in 3 lines.
Then analyze the tone.
Then write a tweet reflecting the tone and summary.

Blog:
LangChain helps developers build powerful LLM apps using chains, agents, and tools. It introduces modular components that can be reused, combined, and debugged easily. This makes it easier to build reliable and interpretable LLM-based workflows.
"""

plain_response = llm.invoke(plain_prompt)
print("\n---  Plain LLM Output (No Chain) ---\n")
print(plain_response.content)

#
#  Structured Chain Approach

# Step 1: Summary
summary_prompt = PromptTemplate.from_template("Summarize this in 3 lines:\n\n{blog}")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# Step 2: Tone
tone_prompt = PromptTemplate.from_template("Analyze the tone of this summary:\n\n{summary}")
tone_chain = LLMChain(llm=llm, prompt=tone_prompt, output_key="tone")

# Step 3: Tweet
tweet_prompt = PromptTemplate.from_template(
    "Write a tweet in a {tone} tone based on the following summary:\n\n{summary}"
)
tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, output_key="tweet")

#  Sequential Chain
chained_pipeline = SequentialChain(
    chains=[summary_chain, tone_chain, tweet_chain],
    input_variables=["blog"],
    output_variables=["summary", "tone", "tweet"],
    verbose=True
)

# Input blog
blog_text = """LangChain helps developers build powerful LLM apps using chains, agents, and tools. It introduces modular components that can be reused, combined, and debugged easily. This makes it easier to build reliable and interpretable LLM-based workflows."""

chain_result = chained_pipeline.invoke({"blog": blog_text})

#  Output
print("\n---  LangChain Structured Output ---\n")
print(" Summary:\n", chain_result["summary"])
print("\n Tone:\n", chain_result["tone"])
print("\n Tweet:\n", chain_result["tweet"])