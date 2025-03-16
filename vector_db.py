import json
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import glog
from typing import Optional, Dict
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv(override=True)

EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "openai")  # Default to OpenAI
VECTOR_DB_PATH = "./chroma_db"
collection_name = "university_notes"

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

if EMBEDDER_TYPE == "openai":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedder = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key)
elif EMBEDDER_TYPE == "huggingface":
    embedder = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.getenv("HF_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
else:
    raise ValueError(f"Unsupported embedder type: {EMBEDDER_TYPE}")

collection = client.get_or_create_collection(name=collection_name,
                                             embedding_function=embedder)


def add_document(doc_id: str,
                 content: str,
                 metadata: Optional[Dict] = None) -> str:
    """
    Adds a document to the vector database if it doesn't already exist.

    This function checks if a document with the given `doc_id` already exists in
    the collection. If it does not exist, the document is added to the database;
    otherwise, it skips adding the document and logs the event.

    Args:
        doc_id (str): A unique identifier for the document.
        content (str): The content of the document to be embedded and added.
        metadata (Optional[Dict], optional): Metadata related to the document (
        e.g., categories, authors). Defaults to None.

    Returns:
        str: A message indicating the success or reason for skipping the document.
    """
    if metadata is None:
        metadata = {}

    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Metadata is not JSON-serializable: {metadata}") from e

    existing_docs = collection.get(ids=[doc_id])
    if not existing_docs["ids"]:
        collection.add(ids=[doc_id], documents=[content], metadatas=[metadata])
        return f"Document {doc_id} added successfully."

    return f"Document {doc_id} already exists in the collection. Skipping embedding."


def query_documents(query_text: str, n_results: int = 5) -> Dict:
    """
    Queries the vector database for documents similar to the given query.

    Args:
        query_text (str): The query text to search for similar documents.
        n_results (int, optional): The number of results to retrieve. Defaults to 5.

    Returns:
        Dict: A dictionary containing the query results, including document IDs and metadata.
    """
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results


def delete_document(doc_id: str) -> str:
    """
    Deletes a document from the vector database by its `doc_id`.

    Args:
        doc_id (str): The unique identifier of the document to be deleted.

    Returns:
        str: A message indicating the success of the deletion operation.
    """
    collection.delete(ids=[doc_id])
    return f"Document {doc_id} deleted successfully."


def list_documents() -> Dict:
    """
    Lists all stored document IDs in the vector database.

    Returns:
        Dict: A dictionary containing all document IDs stored in the collection.
    """
    return collection.get(include=["documents", "embeddings", "metadatas"])


if __name__ == "__main__":
    glog.info(
        "VectorDB module loaded. Run functions directly to manipulate the database."
    )
