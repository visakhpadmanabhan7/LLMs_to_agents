from langchain.chains.router.multi_prompt import MultiPromptChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#hide warnings
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

prompt_1_template = """
You are an expert on animals. Please answer the below query:

{input}
"""

prompt_2_template = """
You are an expert on vegetables. Please answer the below query:

{input}
"""

prompt_infos = [
    {
        "name": "animals",
        "description": "prompt for an animal expert",
        "prompt_template": prompt_1_template,
    },
    {
        "name": "vegetables",
        "description": "prompt for a vegetable expert",
        "prompt_template": prompt_2_template,
    },
]

chain = MultiPromptChain.from_prompts(llm, prompt_infos)
while True:
    query = input("\nAsk your question (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break

    router_output = chain.router_chain.invoke({"input": query})
    destination = router_output["destination"]

    print(f"\ Routed to: {destination}")


    result = chain.invoke({"input": query})

    print("\n🤖 Response:", result["text"] if isinstance(result, dict) else result)