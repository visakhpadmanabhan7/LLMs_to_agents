# tools.py

from langchain.agents import Tool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from langchain_community.tools import DuckDuckGoSearchRun
from db_indexing import get_chroma_client, get_categories, get_persist_dir

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Category-specific tool descriptions
TOOL_DESCRIPTIONS = {
    "sports": (
        "Use this tool to retrieve information from internal cricket documents. "
        "Prefer this over web search for sports questions."
    ),
    "food": (
        "Use this tool to retrieve information from internal food documents. "
        "Includes recipes, ingredients, dish origins, and cooking methods. "
        "Prefer this over web search for food-related questions."
    ),
    "movies": (
        "Use this tool to retrieve information from internal movie documents. "
        "Includes plot summaries, character details, release years, and reviews. "
        "Prefer this over web search for movie-related questions."
    ),
}

def load_tools():
    tools = []
    db = get_chroma_client()
    categories = get_categories()
    persist_dir = get_persist_dir()

    for category in categories:
        collection = db.get_or_create_collection(name=category)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir, vector_store=vector_store
        )
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=3)

        tools.append(Tool(
            name=f"{category.capitalize()}Retriever",
            func=query_engine.query,
            description=TOOL_DESCRIPTIONS.get(
                category,
                f"Useful for answering questions about {category}"
            )
        ))

    tools.append(Tool(
        name="web_search",
        func=DuckDuckGoSearchRun().run,
        description="Use for looking up current events, news, facts, or real-time data. "
                    "Please prioritize internal tools for answering questions. Use web search only if no tool is suitable."
    ))

    tools.append(Tool(
        name="calculator",
        func=calculator,
        description="Use for solving math expressions like addition, multiplication, etc."
    ))

    return tools