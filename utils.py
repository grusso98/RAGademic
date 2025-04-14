from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantic_chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Splits a given text into semantic chunks using a recursive character text splitter.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The max size of each chunk. Defaults to 1500.
        overlap (int, optional): The amount of overlap between consecutive chunks. Defaults to 300.

    Returns:
        List[str]: A list of text chunks.
    """
    if not text: # Handle empty input
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # Common separators
        length_function=len,
        is_separator_regex=False, # Treat separators literally
    )
    return splitter.split_text(text)