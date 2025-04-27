from typing import List

# --- Import for old RecursiveCharacterSplitter---
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    
    # ---Old Recursive Character Splitter---
    """splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # Common separators
        length_function=len,
        is_separator_regex=False, # Treat separators literally
    )
    """
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=95,
    )
    return splitter.split_text(text)
    