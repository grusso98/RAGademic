from unittest.mock import MagicMock

import pytest

from utils import semantic_chunk_text


@pytest.fixture
def mock_huggingface_embeddings_class(mocker):
    # Patch the HuggingFaceEmbeddings class imported in utils.py
    mock_embeddings_class = mocker.patch('utils.HuggingFaceEmbeddings')
    # Return the mock class so tests can check its instantiation if needed
    return mock_embeddings_class

@pytest.fixture
def mock_chunker_and_embeddings_instance(mocker):
    # Create a mock instance for the embeddings model
    mock_embeddings_instance = MagicMock()
    # Patch the 'embeddings' instance at the module level in utils.py
    # This ensures that when semantic_chunk_text uses 'embeddings', it gets this mock
    mocker.patch('utils.embeddings', mock_embeddings_instance)

    # Patch the SemanticChunker class imported in utils.py
    mock_chunker_class = mocker.patch('utils.SemanticChunker')
    # Create a mock instance that will be returned when the SemanticChunker class is called (instantiated)
    mock_chunker_instance = MagicMock()
    # Configure the mock class to return the mock instance when called
    mock_chunker_class.return_value = mock_chunker_instance

    # Return the mock chunker instance, the mocked embeddings instance, AND the mock chunker class
    return mock_chunker_instance, mock_embeddings_instance, mock_chunker_class

def test_semantic_chunk_text_empty_string(mock_chunker_and_embeddings_instance):
    """Tests splitting an empty string."""
    # Ensure the SemanticChunker is not called for empty input
    mock_chunker_instance, _, mock_chunker_class = mock_chunker_and_embeddings_instance

    text = ""
    chunks = semantic_chunk_text(text)

    assert chunks == []
    mock_chunker_class.assert_not_called() # SemanticChunker class should not be instantiated
    mock_chunker_instance.split_text.assert_not_called() # split_text method should not be called


def test_semantic_chunk_text_calls_semantic_chunker(
    mock_chunker_and_embeddings_instance # Use the fixture providing the mocks
):
    """
    Tests that SemanticChunker is initialized correctly with the module-level embeddings instance
    and its split_text method is called.
    """
    mock_chunker_instance, mock_embeddings_instance, mock_chunker_class = mock_chunker_and_embeddings_instance

    # Simulate the return value of SemanticChunker's split_text
    mock_split_text_result = ["chunk1", "chunk2", "chunk3"]
    mock_chunker_instance.split_text.return_value = mock_split_text_result

    text = "This is a sentence. This is another sentence. And a third one."
    # Pass chunk_size and overlap, though SemanticChunker ignores them
    chunk_size = 100
    overlap = 20

    chunks = semantic_chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Assert that the mocked SemanticChunker CLASS was called (instantiated)
    mock_chunker_class.assert_called_once_with(
        embeddings=mock_embeddings_instance, # Assert it was called with the mocked instance
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=95,
    )

    # Assert that the split_text method of the mocked SemanticChunker INSTANCE was called
    mock_chunker_instance.split_text.assert_called_once_with(text)

    # Assert that the function returned the result from the mocked split_text call
    assert chunks == mock_split_text_result