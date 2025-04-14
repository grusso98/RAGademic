import hashlib
import re
from typing import Dict, List, Optional, Tuple

import glog
import ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults

from utils import semantic_chunk_text
from vector_db import add_document

_CHECKER_MODEL_ID = "llama3.2:latest"
_CHECKER_SYSTEM_PROMPT = "You are an AI assistant evaluating context relevance."
_CHECKER_PROMPT_TEMPLATE = """
Analyze the Provided Context based on the User Query.
Does the context contain **relevant** information to directly address the main points of the query?
Ignore minor details or completeness, just focus on core relevance.

User Query:
{query}

Provided Context:
{context}

Based **only** on the context provided, is the information relevant? Answer with a single word: **Relevant** or **Insufficient**.
Answer:
"""
_WEB_SEARCH_NUM_RESULTS = 5
_SCRAPING_TIMEOUT = 15 
_MIN_SCRAPED_CONTENT_LENGTH = 100

def is_context_sufficient(query: str, context: str) -> bool:
    """
    Uses a local LLM (llama3.2) to check if the context is relevant for the query.
    Uses a revised prompt and more robust response parsing.

    Args:
        query: The user's query.
        context: The initially retrieved context from the knowledge base.

    Returns:
        True if the context is deemed relevant, False otherwise.
    """
    
    if not context or len(context.strip()) < 50: 
        glog.warning("Context is empty or very short. Deeming insufficient.")
        return False
    if context == "No relevant documents found." or context == "Error retrieving context from documents.":
        glog.info("Context indicates retrieval failure. Deeming insufficient.")
        return False

    prompt = _CHECKER_PROMPT_TEMPLATE.format(query=query, context=context)
    glog.info(f"Checking context sufficiency for query '{query}' using {_CHECKER_MODEL_ID}...")
    glog.debug(f"Checker Prompt Context Sample: {context[:500]}...") 

    try:
        response = ollama.chat(
            model=_CHECKER_MODEL_ID,
            messages=[
                {"role": "system", "content": _CHECKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            options={'temperature': 0.1} # Keep low temp for consistency
        )

        if response and 'message' in response and 'content' in response['message']:
            decision_text = response['message']['content'].strip().lower()
            glog.info(f"Sufficiency checker raw response: '{decision_text}'")

            if re.search(r'\brelevant\b', decision_text):
                glog.info("Checker response indicates RELEVANT.")
                return True
            elif re.search(r'\binsufficient\b', decision_text):
                glog.info("Checker response indicates INSUFFICIENT.")
                return False
            else:
                # Fallback: If keywords aren't found, maybe default to insufficient or try basic yes/no
                glog.warning(f"Checker response ('{decision_text}') did not contain clear 'Relevant' or 'Insufficient'. Defaulting to INSUFFICIENT.")
                # You could add a check for 'yes' here as another fallback if needed
                # if decision_text.startswith('yes'): return True
                return False
        else:
            glog.error("Checker model returned an invalid response structure.")
            return False 

    except Exception as e:
        glog.error(f"Error calling checker model ({_CHECKER_MODEL_ID}): {e}", exc_info=True)
        return False 


def perform_web_search(query: str, num_results: int = _WEB_SEARCH_NUM_RESULTS) -> List[Dict]:
    """
    Performs a web search using DuckDuckGo and returns structured results.
    Includes enhanced logging and stricter link parsing/cleaning.
    """
    glog.info(f"Performing web search for query: '{query}' with {num_results} results")
    results = []
    try:
        search_tool = DuckDuckGoSearchResults(num_results=num_results)
        search_output_raw = search_tool.run(query)

        glog.debug(f"Raw DDG Search Output: {search_output_raw}")

        if not search_output_raw or search_output_raw.strip() == "No good DuckDuckGo Search Result was found":
             glog.warning("DuckDuckGo search returned no results.")
             return []

        pattern = re.compile(
            r"snippet:\s?(.*?),?\s?title:\s?(.*?),?\s?link:\s?(https?://[^\s,\]]+)",
            re.IGNORECASE | re.DOTALL
        )
        matches = pattern.findall(search_output_raw)

        if not matches:
             glog.warning(f"Could not parse DDG results string using regex. Raw output sample: {search_output_raw[:300]}...")
             return []

        for match in matches:
            snippet = match[0].strip()
            title = match[1].strip()
            link = match[2].strip()

            if link:
                results.append({
                    "title": title if title else "Untitled",
                    "link": link, 
                    "snippet": snippet if snippet else "No snippet available."
                })
            else:
                 glog.warning(f"Skipping result with invalid link extracted: link='{match[2]}', title='{title}'")


        glog.info(f"Successfully parsed {len(results)} web search results.")
        return results[:num_results]

    except Exception as e:
        glog.error(f"Error during web search or parsing: {e}", exc_info=True)
        return []

def scrape_web_content(url: str) -> Optional[str]:
    """
    Scrapes the main text content from a given URL using LangChain's WebBaseLoader.

    Args:
        url: The URL to scrape.

    Returns:
        The extracted text content, or None if scraping fails.
    """
    glog.info(f"Attempting to scrape content from: {url} using WebBaseLoader")
    try:
        # Configure WebBaseLoader
        # You might need to install 'unstructured' or 'beautifulsoup4' if not already present
        # WebBaseLoader uses BeautifulSoup4 by default
        loader = WebBaseLoader(
            web_paths=[url],
            # Optional: Add headers if needed, though WBL might handle some internally
            # header_template = {
            #     'User-Agent': 'Mozilla/5.0 ...'
            # }
            # Optional: Configure the BS4 parser if needed
            # bs_kwargs=dict(parse_only=bs4.SoupStrainer(...))
        )
        loader.requests_kwargs = {'timeout': _SCRAPING_TIMEOUT} # Set timeout

        # Load the document(s)
        documents = loader.load()

        if not documents:
            glog.warning(f"WebBaseLoader returned no documents for URL: {url}")
            return None

        # Combine content from potentially multiple documents (though usually one for one URL)
        full_content = "\n".join([doc.page_content for doc in documents if doc.page_content])

        if not full_content.strip():
             glog.warning(f"WebBaseLoader returned empty content for URL: {url}")
             return None

        glog.info(f"WebBaseLoader successfully scraped ~{len(full_content)} characters from {url}")
        return full_content

    except Exception as e:
        # Catch specific exceptions if needed (e.g., ConnectionError, Timeout)
        glog.error(f"Error scraping content with WebBaseLoader from {url}: {e}", exc_info=True)
        return None


def search_scrape_and_ingest(query: str, num_results: int = _WEB_SEARCH_NUM_RESULTS) -> Tuple[str, List[Dict]]:
    """
    Performs web search, scrapes content, adds to vector DB,
    and returns the combined scraped text AND metadata of added chunks.

    Args:
        query: The user query to search for.
        num_results: Number of search results to process.

    Returns:
        A tuple containing:
            - str: Combined text content from successfully scraped pages.
            - List[Dict]: A list of metadata dictionaries for chunks successfully added to the DB.
    """
    search_results = perform_web_search(query, num_results)
    if not search_results:
        glog.warning("search_scrape_and_ingest: No search results found.")
        return "", [] 

    combined_scraped_text = ""
    processed_urls = set()
    added_chunks_metadata = [] 
    added_chunks_count = 0

    glog.info(f"Attempting to scrape up to {len(search_results)} URLs...")
    for i, result in enumerate(search_results):
        url = result.get('link')
        title = result.get('title', 'Unknown Title')
        glog.info(f"Processing search result #{i+1}: {title} ({url})")

        if not url or not url.startswith('http') or url in processed_urls:
            glog.warning(f"Skipping invalid or duplicate URL: {url}")
            continue

        processed_urls.add(url)
        content = scrape_web_content(url) # Uses WebBaseLoader

        if content and len(content) >= _MIN_SCRAPED_CONTENT_LENGTH:
            glog.info(f"Successfully scraped sufficient content ({len(content)} chars) from {url}")
            # Append content for the final context string
            combined_scraped_text += f"\n\n--- Content from {url} ---\n{content}"

            # Add scraped content to knowledge base
            try:
                chunks = semantic_chunk_text(content, chunk_size=1500, overlap=300)
                if not chunks:
                     glog.warning(f"Content from {url} resulted in zero chunks after splitting.")
                     continue

                glog.info(f"Splitting content from {url} into {len(chunks)} chunks.")
                url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
                safe_query_part = re.sub(r'\W+', '_', query[:20])
                doc_filename = f"web_{url_hash}_{safe_query_part}.txt"

                for chunk_index, chunk in enumerate(chunks): # Use enumerate for index
                    if not chunk.strip():
                        continue

                    chunk_id = f"web_{url_hash}_q{safe_query_part}_part{chunk_index}"
                    metadata = {
                        "source": "web",
                        "category": f"web_search_{safe_query_part}",
                        "original_query": query,
                        "url": url,
                        "title": title[:150],
                        "chunk_index": chunk_index, # Store chunk index
                        "filename": doc_filename
                    }
                    # Add document to vector DB
                    status = add_document(chunk_id, chunk, metadata)
                    glog.debug(f"Adding chunk {chunk_id}: {status}")
                    if "added successfully" in status:
                        added_chunks_count += 1
                        # Add metadata to the list to be returned
                        added_chunks_metadata.append(metadata)

            except Exception as e:
                glog.error(f"Error chunking or adding document for {url}: {e}", exc_info=True)
        elif content:
             glog.warning(f"Scraped content from {url} was too short ({len(content)} chars), skipping ingestion.")
        else:
            glog.warning(f"Scraping failed for URL: {url}")

    glog.info(f"Finished web search & scrape. Added {added_chunks_count} web chunks to DB.")
    # Return both the combined text and the list of metadata
    return combined_scraped_text.strip(), added_chunks_metadata
