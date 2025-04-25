import pytest

from utils import semantic_chunk_text


def test_semantic_chunk_text_basic():
    """Tests basic text splitting with default parameters."""
    text = "This is the first sentence. This is the second sentence. This is the third sentence, which is a bit longer."
    chunks = semantic_chunk_text(text, chunk_size=50, overlap=10)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # Check if chunks are within the expected size (considering overlap)
    for chunk in chunks:
        assert len(chunk) <= 50 + 10 # Allow for overlap content

def test_semantic_chunk_text_empty_string():
    """Tests splitting an empty string."""
    text = ""
    chunks = semantic_chunk_text(text)
    assert chunks == []

def test_semantic_chunk_text_smaller_than_chunk_size():
    """Tests splitting text smaller than the chunk size."""
    text = "This is a short text."
    chunks = semantic_chunk_text(text, chunk_size=100, overlap=20)
    assert chunks == ["This is a short text."]

def test_semantic_chunk_text_with_different_separators():
    """Tests splitting text with different separators."""
    text = "Line1\n\nLine2.\nLine3. Another sentence."
    chunks = semantic_chunk_text(text, chunk_size=20, overlap=5)
    assert isinstance(chunks, list)
    assert len(chunks) > 1 # Expecting multiple chunks
