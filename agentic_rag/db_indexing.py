# db_indexing.py

import chromadb
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

from llama_index.vector_stores.chroma import ChromaVectorStore
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

input_base_dir = Path("./data")
categories = {
    folder.name: folder
    for folder in input_base_dir.iterdir()
    if folder.is_dir()
}

db = chromadb.PersistentClient(path="./chroma_db")
persist_dir = "./chroma_index"

def check_index_exists():
    existing_collections = [c.name for c in db.list_collections()]
    missing_indexes = [name for name in categories if name not in existing_collections]

    if missing_indexes:
        print(f"Missing indexes for: {', '.join(missing_indexes)}. Creating...")
        create_index()
    else:
        print(" All vector indexes already exist. Skipping creation.")

def create_index():
    for name, folder_path in categories.items():
        print(f" Processing category: {name} from folder: {folder_path}")
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            continue

        documents = SimpleDirectoryReader(input_dir=folder_path).load_data()
        documents = [doc for doc in documents if doc and doc.text.strip()]
        if not documents:
            print(f"No valid documents found in {folder_path}. Skipping.")
            continue

        collection = db.get_or_create_collection(name=name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        index.storage_context.persist(persist_dir=persist_dir)
        print(f" Successfully indexed category: {name}")

def get_categories():
    return categories

def get_chroma_client():
    return db

def get_persist_dir():
    return persist_dir