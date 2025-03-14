import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import glog

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "university_notes"

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_embedder = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key)

collection = client.get_or_create_collection(name=collection_name, embedding_function=openai_embedder)

def add_document(doc_id: str, content: str, metadata: dict = {}):
    """Adds a document to the vector database."""
    collection.add(ids=[doc_id], documents=[content], metadatas=[metadata])
   
def query_documents(query_text: str, n_results: int = 5):
    """Queries the vector database for similar documents."""
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results

def delete_document(doc_id: str):
    """Deletes a document from the vector database."""
    collection.delete(ids=[doc_id])

def list_documents():
    """Lists all stored document IDs."""
    return collection.peek()

if __name__ == "__main__":
    print("VectorDB module loaded. Run functions directly to manipulate the database.")
