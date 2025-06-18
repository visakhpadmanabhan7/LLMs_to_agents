import chromadb
from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.readers.file import PDFReader

import os
from pathlib import Path

reader = PDFReader()
documents = []

pdf_folder = Path("../data")
for pdf_file in pdf_folder.glob("*.pdf"):
    docs = reader.load_data(file=pdf_file)
    documents.extend(docs)
# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Check for existing index directory, load or create and persist index


PERSIST_DIR = "./chroma_index"

if os.path.exists(PERSIST_DIR):
    # Load from disk
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR, vector_store=vector_store
    )
    index = load_index_from_storage(storage_context)
    print("Index loaded from disk.")

else:
    # Create and persist new index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index created and persisted.")
print('--' * 50)
prompt1="Which team had the worst catching efficiency till the mid of IPL 2025?"
prompt2='Which among the three Indian wicket keepers has a better strike rate at the start in IPL 2024?'
prompt3='Which bowler type succeeds the most against SuryaKumar Yadav?'
prompts=[prompt1, prompt2, prompt3]

#Define LLM
llm = OpenAI(model="gpt-4.1-mini")

for prompt in prompts:
    print(f"Prompt:\n{prompt}\n")

    # Direct LLM completion (without RAG)
    response = llm.complete(prompt)
    print("Response without RAG:\n", response)


    # Setting up RAG components
    retriever = index.as_retriever(similarity_top_k=3)
    synthesizer = get_response_synthesizer(llm=llm)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )

    # Query with RAG
    print("Response with RAG:")
    rag_response = query_engine.query(prompt)
    print(rag_response)

    print('-' * 100)
