import warnings

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
warnings.filterwarnings("ignore")
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

prompt = """
Summarize the following blog post in 3 lines. 
Then analyze its tone. 
Then write a tweet that reflects the tone and key points of the summary.

Blog:
In this post, we explore how LangChain agents use the ReAct framework to reason step-by-step.
"""

output = llm.invoke(prompt)
print(output.content)