import os
from unittest.mock import MagicMock, patch

import pytest


# Mocking environment variables before importing vector_db
@pytest.fixture(autouse=True)
def mock_env_vars(mocker):
    mocker.patch.dict(os.environ, {'EMBEDDER_TYPE': 'local', 'OPENAI_API_KEY': 'fake_openai_key', 'HF_TOKEN': 'fake_hf_token'})

from vector_db import (add_document, delete_document, list_documents,
                       query_documents)


# Mock the ChromaDB PersistentClient and collection for all tests in this file
@pytest.fixture(autouse=True)
def mock_chromadb(mocker):
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    
    mocker.patch('vector_db.chromadb.PersistentClient', return_value=mock_client)
    mocker.patch('vector_db.collection', mock_collection) # Patch the module-level collection instance
    mocker.patch('vector_db.client', mock_client) # Patch the module-level client instance

    # Also mock the embedder function creation if needed for different types
    mock_embedder_func = MagicMock()
    mocker.patch('vector_db.embedding_functions.SentenceTransformerEmbeddingFunction', return_value=mock_embedder_func)
    mocker.patch('vector_db.embedding_functions.OpenAIEmbeddingFunction', return_value=mock_embedder_func)
    mocker.patch('vector_db.embedding_functions.HuggingFaceEmbeddingFunction', return_value=mock_embedder_func)
    mocker.patch('vector_db.embedder', mock_embedder_func) # Patch the module-level embedder instance

    return mock_collection # Return the mock collection for easier assertions in tests


def test_add_document_success(mock_chromadb):
    """Tests successfully adding a new document."""
    mock_chromadb.get.return_value = {"ids": []} # Document does not exist
    doc_id = "doc123"
    content = "This is the content of the document."
    metadata = {"category": "test"}

    result = add_document(doc_id, content, metadata)

    mock_chromadb.get.assert_called_once_with(ids=[doc_id])
    mock_chromadb.add.assert_called_once_with(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata]
    )
    assert result == f"Document {doc_id} added successfully."

def test_add_document_already_exists(mock_chromadb):
    """Tests adding a document that already exists."""
    mock_chromadb.get.return_value = {"ids": ["doc123"]} # Document exists
    doc_id = "doc123"
    content = "This is the content of the document."
    metadata = {"category": "test"}

    result = add_document(doc_id, content, metadata)

    mock_chromadb.get.assert_called_once_with(ids=[doc_id])
    mock_chromadb.add.assert_not_called() # Add should not be called
    assert result == f"Document {doc_id} already exists in the collection. Skipping embedding."

def test_add_document_invalid_metadata(mock_chromadb):
    """Tests adding a document with non-JSON-serializable metadata."""
    doc_id = "doc124"
    content = "Content"
    metadata = {"object": object()} # Non-JSON-serializable object

    with pytest.raises(ValueError, match="Metadata is not JSON-serializable"):
        add_document(doc_id, content, metadata)

    mock_chromadb.get.assert_not_called()
    mock_chromadb.add.assert_not_called()

def test_query_documents(mock_chromadb):
    """Tests querying documents."""
    query_text = "search query"
    n_results = 3
    mock_results = {
        'ids': [['doc1', 'doc2', 'doc3']],
        'documents': [['content1', 'content2', 'content3']],
        'metadatas': [{'meta1': 'data1'}, {'meta2': 'data2'}, {'meta3': 'data3'}],
        'distances': [[0.1, 0.2, 0.3]]
    }
    mock_chromadb.query.return_value = mock_results

    results = query_documents(query_text, n_results)

    mock_chromadb.query.assert_called_once_with(
        query_texts=[query_text],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'],
    )
    assert results == mock_results

def test_delete_document(mock_chromadb):
    """Tests deleting a document."""
    doc_id = "doc456"
    result = delete_document(doc_id)

    mock_chromadb.delete.assert_called_once_with(ids=[doc_id])
    assert result == f"Document {doc_id} deleted successfully."

def test_list_documents(mock_chromadb):
    """Tests listing documents."""
    mock_document_list = {
        'ids': ['doc1', 'doc2'],
        'documents': ['content1', 'content2'],
        'embeddings': [[0.1, 0.2], [0.3, 0.4]],
        'metadatas': [{'meta1': 'data1'}, {'meta2': 'data2'}]
    }
    mock_chromadb.get.return_value = mock_document_list

    result = list_documents()

    mock_chromadb.get.assert_called_once_with(include=["documents", "embeddings", "metadatas"])
    assert result == mock_document_list
