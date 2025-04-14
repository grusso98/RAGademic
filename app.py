import glob
import os
import random
import re
import urllib
import xml.etree.ElementTree as ET
from typing import Dict, List

import glog
import gradio as gr
import numpy as np
import ollama
import plotly.graph_objs as go
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from openai import OpenAI
from sklearn.manifold import TSNE

from agent import is_context_sufficient, search_scrape_and_ingest
from memory_manager import MemoryManager
from utils import semantic_chunk_text
from vector_db import add_document, collection, list_documents, query_documents

load_dotenv(override=True)

_SYSTEM_PROMPT: str ="""
You are a helpful university tutor, students will ask you questions about their 
books, studies or articles. You may be provided with additional context for some questions.
If provided, answer based on the context. If you do not know something, clearly
state it. You may also be provided with a summary of past conversations. Use this
summary to maintain context across sessions.
"""
_BASE_PROMPT: str = """
Use the following information to answer the question:

Long-term Memory Summary:
{memory_summary}

Context from Documents and Web Search (if applicable):
{context}

Current Conversation History (most recent turns):
{chat_history}

Question: {query}

Answer:
"""

_MODEL_SIZE: str = os.getenv("MODEL_SIZE", "4b")
_KB_FOLDER: str = "knowledge_base/"

DEFAULT_SUMMARY_MODEL = "llama3.2:latest" 
memory_manager = MemoryManager(model_identifier=DEFAULT_SUMMARY_MODEL)

def load_notes() -> str:
    """
    Loads notes from the knowledge base folder and adds them to the vector DB.
    Processes various document formats using MarkItDown.

    Returns:
        str: Status message indicating the number of processed files.
    """
    _KB_FOLDER = "./knowledge_base"
    loaded_count = 0
    md_converter = MarkItDown()

    for category in os.listdir(_KB_FOLDER):
        category_path = os.path.join(_KB_FOLDER, category)
        if os.path.isdir(category_path):
            for file_path in glob.glob(f"{category_path}/*"):
                glog.info(f"Processing file: {file_path}")
                try:
                    result = md_converter.convert(file_path)
                    content = result.text_content

                    if not content.strip():
                        glog.warning(f"No text extracted from {file_path}, skipping.")
                        continue

                    chunks = semantic_chunk_text(content, chunk_size=2000, overlap=400)
                    doc_id_base = os.path.basename(file_path)

                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id_base}_part{i}"
                        add_document(
                            chunk_id, chunk, {
                                "category": category,
                                "file_path": os.path.abspath(file_path),
                                "chunk_index": i,
                                "filename": doc_id_base
                            })
                    loaded_count += 1
                    glog.info(f"Added {len(chunks)} chunks for {doc_id_base}")
                except Exception as e:
                    glog.error(f"Failed to process {file_path}: {e}")
    return f"Finished loading notes. Processed {loaded_count} files."


def search_and_ingest_papers(query: str, max_results: int = 5) -> str:
    """
    Searches arXiv for papers related to the query and ingests abstracts into the vector DB.

    Args:
        query (str): The user query/topic to search.
        max_results (int): Number of papers to fetch.

    Returns:
        str: Summary of ingestion results.
    """
    glog.info(f"Searching arXiv for query: {query}")
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}"
    try:
        response = requests.get(url, timeout=20) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        glog.error(f"Failed to fetch papers from arXiv: {e}")
        return f"Failed to fetch papers from arXiv: {e}"


    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
         glog.error(f"Failed to parse arXiv XML response: {e}")
         return "Failed to parse arXiv response."

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    if not entries:
        return "No papers found matching the query."

    count = 0
    skipped = 0
    for entry in entries:
        try:
            title_elem = entry.find("atom:title", ns)
            abstract_elem = entry.find("atom:summary", ns)
            id_elem = entry.find("atom:id", ns)
            link_elem = entry.find("atom:link[@rel='alternate'][@type='text/html']", ns) 
            
            if title_elem is None or abstract_elem is None or id_elem is None or link_elem is None:
                glog.warning("Skipping entry due to missing essential fields.")
                skipped += 1
                continue

            title = title_elem.text.strip()
            abstract = abstract_elem.text.strip().replace("\n", " ") 
            arxiv_id_url = id_elem.text.strip()
            arxiv_id = arxiv_id_url.split('/')[-1].split('v')[0]
            link = link_elem.attrib['href']


            glog.info(f"Ingesting paper: {arxiv_id} - {title}")
            
            full_text = f"Title: {title}\nAbstract: {abstract}\nLink: {link}"
            chunks = semantic_chunk_text(full_text, chunk_size=1500, overlap=300) 

            for i, chunk in enumerate(chunks):
                chunk_id = f"arxiv_{arxiv_id}_part{i}"
                add_document(chunk_id, chunk, {
                    "source": "arXiv",
                    "category": f"arxiv_{query.lower().replace(' ', '_')}",
                    "query": query,
                    "arxiv_id": arxiv_id,
                    "link": link,
                    "title": title, 
                    "chunk_index": i,
                    "filename": f"{arxiv_id}.pdf"
                })

            count += 1
        except Exception as e:
            glog.error(f"Error processing arXiv entry: {e}")
            skipped +=1

    result_message = f"Ingested {count} papers from arXiv for query '{query}'."
    if skipped > 0:
        result_message += f" Skipped {skipped} entries due to errors or missing data."
        
    return result_message

def query_rag(
    query: str,
    model: str,
    history: List[List[str]],
    enable_local_rag: bool, 
    enable_web_agent: bool, 
    memory_summary: str
    ):
    """
    Queries the RAG system, incorporating long-term memory summary.

    Args:
        query (str): The user's query.
        model (str): The model identifier (e.g., 'llama3.2:latest', 'gpt-4o-mini').
        history (List[List[str]]): The *current session's* chat history.
        enable_local_rag (bool): Whether to use RAG context.
        enable_web_agent (bool): Whether to activate the web search agent.
        memory_summary (str): The summary of past conversations.

    Yields:
        str: Chunks of the generated response.
    """
    context_for_llm = ""
    final_source_links = []
    web_search_attempted = False 

    if enable_local_rag:
        glog.info(f"Local RAG enabled. Querying vector DB for: '{query}'")
        local_context = ""
        local_sources_metadata = []
        try:
            # --- 1. Attempt Local Retrieval ---
            results = query_documents(query, n_results=5) 
            if results and results.get("documents"):
                context_docs = []
                doc_lists = results.get("documents", [])
                if doc_lists and isinstance(doc_lists[0], list): context_docs.extend(doc for sublist in doc_lists for doc in sublist)
                else: context_docs = doc_lists
                local_context = "\n\n".join(filter(None, context_docs))

                meta_lists = results.get("metadatas", [])
                if meta_lists:
                    if isinstance(meta_lists[0], list): local_sources_metadata.extend(meta for sublist in meta_lists for meta in sublist if meta)
                    else: local_sources_metadata.extend(meta for meta in meta_lists if meta)
                glog.info(f"Retrieved {len(context_docs)} context snippets from local DB.")
            else:
                glog.info("No relevant documents found initially in vector DB.")
                local_context = "" 

            # --- 2. Decide Action Based on Local Context and Toggles ---
            if local_context:
                # --- Local Context Found ---
                use_local_context = True # Assume we use local initially
                if enable_web_agent:
                    # Web agent is enabled, check sufficiency
                    glog.info("Web agent enabled, checking local context sufficiency...")
                    context_sufficient = is_context_sufficient(query, local_context)
                    glog.info(f"Local context sufficient check result: {context_sufficient}")
                    if not context_sufficient:
                        # Insufficient and web agent is ON -> Try web search
                        use_local_context = False # Signal to use web search instead
                        glog.info("Local context insufficient, will attempt web search.")
                    else:
                        glog.info("Local context sufficient, proceeding with local.")
                else:
                    # Web agent disabled, must use local context if found
                    glog.info("Web agent disabled, using local context.")

                if use_local_context:
                    context_for_llm = local_context
                    final_source_links = generate_source_links(local_sources_metadata, "local")
                else:
                    # Trigger web search (because local insufficient and agent enabled)
                    web_search_attempted = True
                    # (Web search logic moved below to avoid duplication)

            else:
                # --- No Local Context Found ---
                glog.info("No local context found.")
                if enable_web_agent:
                    # Try web search directly
                    glog.info("Web agent enabled, attempting web search directly.")
                    web_search_attempted = True
                    # (Web search logic below will handle this)
                else:
                    # No local, web agent disabled -> Report no context
                    glog.info("Web agent disabled, no context available.")
                    context_for_llm = "No relevant context found in local knowledge base."
                    final_source_links = ["[Info] No local context found."]

            # --- 3. Perform Web Search (if flagged) ---
            if web_search_attempted:
                glog.info("Attempting agentic web search...")
                web_context_text, web_sources_metadata = search_scrape_and_ingest(query)
                if web_context_text:
                    glog.info(f"Using context from web search ({len(web_context_text)} chars).")
                    # Use web context (replace any previous local context assignment)
                    context_for_llm = f"--- Context from Web Search ---\n{web_context_text}"
                    final_source_links = generate_source_links(web_sources_metadata, "web")
                else:
                    glog.warning("Web search did not yield usable context.")
                    # Determine fallback message based on whether local was tried
                    if local_context: # We tried local first, then web failed
                         context_for_llm = "Relevant context not found locally. Web search failed."
                         final_source_links = ["[Info] Local context insufficient, web search failed."]
                    else: # No local, web failed too
                         context_for_llm = "No relevant context found locally or from web search."
                         final_source_links = ["[Web Search Failed] No context found."]

        except Exception as e:
            glog.error(f"Error during RAG retrieval or agentic step: {e}", exc_info=True)
            context_for_llm = "Error processing context retrieval."
            final_source_links = ["[Error] Error retrieving sources."]

    else: # Local RAG Disabled
        glog.info("Local RAG is disabled by user.")
        context_for_llm = "RAG Disabled"
        final_source_links = ["[Info] RAG Disabled."]


    # --- Final Source Text Construction ---
    if final_source_links:
         if final_source_links[0].startswith("["):
              sources_text = f"\n\n**Sources:**\n- {final_source_links[0]}"
         else:
              sources_text = "\n\n**Sources:**\n" + "\n".join(final_source_links)
    else:
         sources_text = "\n\n**Sources:**\n- No relevant context found."
    # --- End Source Text Construction ---

    # --- Prepare Prompt for LLM ---
    history_limit = 5
    recent_history = history[-history_limit:]
    chat_history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in recent_history]) if recent_history else "No current session history."

    prompt = _BASE_PROMPT.format(
        memory_summary=memory_summary if memory_summary else "No long-term memory summary available.",
        context=context_for_llm,
        chat_history=chat_history_text,
        query=query
    )
    glog.debug(f"Final prompt context length: {len(context_for_llm)}")

    # --- LLM Call ---
    try:
        provider = "ollama" 
        if "gpt" in model:
            provider = "openai"
        elif model not in ["llama3.2", "gemma3"]: 
             glog.warning(f"Model '{model}' not explicitly listed as Ollama/OpenAI, assuming Ollama.")
             provider = "ollama"

        if provider == "ollama":
            llm_model_id = model
            if model == "gemma3":
                llm_model_id = f"{model}:{_MODEL_SIZE}" 
            if ':' not in llm_model_id:
                llm_model_id += ":latest"

            glog.info(f"Streaming response from Ollama model: {llm_model_id}")
            response = ollama.chat(
                model=llm_model_id,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT}, 
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            full_response_content = ""
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    content_piece = chunk["message"]["content"]
                    full_response_content += content_piece
                    yield content_piece     

        elif provider == "openai":
            glog.info(f"Streaming response from OpenAI model: {model}")
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                glog.error("OPENAI_API_KEY environment variable not set.")
                yield "Error: OpenAI API key not configured."
                yield sources_text 
                return

            openai_client = OpenAI(api_key=openai_api_key)
            
            openai_base_url = os.getenv("OPENAI_API_BASE")
            if openai_base_url:
                openai_client.base_url = openai_base_url

            stream = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            full_response_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    full_response_content += content_piece
                    yield content_piece 
            
        else:
             glog.error(f"Model provider determination failed for '{model}'.")
             yield f"Error: Cannot determine how to handle model '{model}'."
             yield sources_text
             return 

        yield sources_text

    except Exception as e:
        glog.exception(f"Error during LLM call with model {model}:") 
        yield f"\n\nAn error occurred while generating the response: {e}" 
        yield sources_text 

def generate_source_links(metadata_list: List[Dict], source_type: str = "local") -> List[str]:
    """Generates formatted markdown links for citations from metadata.
    Args:
        metadata_list: A list of dictionaries containing metadata.
        source_type: The type of source (default: "local").

    Returns:
        A list of strings, where each string is a source link.
    """
    if not metadata_list:
        glog.warning("generate_source_links called with empty metadata_list.")
        return []

    source_links = []
    used_docs_info = {} 

    for idx, meta in enumerate(metadata_list):
        if not isinstance(meta, dict):
            glog.warning(f"Skipping item {idx} because it's not a dictionary: {type(meta)} - {meta}")
            continue
    
        url = meta.get("url")
        file_path = meta.get("file_path")
        arxiv_id = meta.get("arxiv_id")

        url = str(url).strip() if url is not None else None
        file_path = str(file_path).strip() if file_path is not None else None
        arxiv_id = str(arxiv_id).strip() if arxiv_id is not None else None

        grouping_key = url if (url and url.startswith('http')) else file_path if file_path else (f"arxiv_{arxiv_id}" if arxiv_id else None)

        if not grouping_key:
            glog.warning(f"Could not determine valid grouping key for meta item {idx}: {meta}")
            continue

        if grouping_key not in used_docs_info:
            raw_title = meta.get("title") or meta.get("filename") or grouping_key.split('/')[-1] or "Unknown Source"
            title = str(raw_title)
            title = re.sub(r'[\[\]\(\)\n\r]', '', title).strip() 
            if len(title) > 70: title = title[:67].strip() + "..."
            if not title: title = "Source" 
            link_url = "#" 
            if url and url.startswith('http'):
                link_url = url.splitlines()[0].strip() 
            elif arxiv_id:
                 link_url = f"https://arxiv.org/abs/{arxiv_id}"
            elif file_path:
                 try:
                     quoted_path = urllib.parse.quote(file_path)
                     link_url = f"file://{quoted_path}"
                 except Exception as e:
                     glog.warning(f"Failed to quote file path '{file_path}': {e}")
                     link_url = "#"

            prefix = ""
            meta_source = meta.get("source", "").lower()
            if meta_source == "web" or (url and not file_path and not arxiv_id):
                prefix = "(Web) "
            elif meta_source == "local" or file_path:
                prefix = "(Local) "
            elif meta_source == "arxiv" or arxiv_id:
                prefix = "(arXiv) "

            display_title = f"{prefix}{title}"
            used_docs_info[grouping_key] = {"title": display_title, "url": link_url}
            glog.debug(f"Stored info for key '{grouping_key}': Title='{display_title}', URL='{link_url}'")

    for i, info in enumerate(used_docs_info.values()):
         final_title = info.get('title', 'Link')
         final_url = info.get('url', '#')
         link_string = f"[{i+1}] [{final_title}]({final_url})"
         source_links.append(link_string)

    glog.info(f"Generated {len(source_links)} source link strings.")
    return source_links

def visualize_embeddings_interactive() -> go.Figure | str: 
    """
    Visualizes document embeddings interactively using Plotly and colors
    them based on their categories.

    Returns:
        go.Figure | str: Plotly figure object or an error message string.
    """
    try:
        documents = list_documents()
        glog.info(f"Visualizing embeddings for {len(documents.get('ids', []))} documents.")

        if not documents or not documents.get("ids"):
            glog.warning("No documents found for visualization.")
            return "No documents available for visualization."

        embeddings = np.array(documents.get('embeddings', []))
        doc_ids = documents.get('ids', [])
        metadatas = documents.get('metadatas', [])

        if embeddings.ndim != 2 or embeddings.shape[0] != len(doc_ids):
             glog.error(f"Embeddings shape mismatch or invalid: {embeddings.shape}, expected ({len(doc_ids)}, N)")
             return "Error: Embeddings data is inconsistent or missing."
             
        if embeddings.shape[0] < 2:
            glog.warning("Not enough documents (less than 2) to generate t-SNE visualization.")
            return "Not enough documents (at least 2 required) to generate visualization."

        categories = [
            metadata.get('category', 'Unknown') if metadata else 'Unknown'
            for metadata in metadatas
        ]

        unique_categories = sorted(list(set(categories)))
        try:
            import plotly.express as px
            color_sequence = px.colors.qualitative.Plotly 
            category_color_map = {cat: color_sequence[i % len(color_sequence)] for i, cat in enumerate(unique_categories)}
        except ImportError:
            glog.warning("plotly.express not available for color sequences, using random colors.")
            random.seed(42) 
            category_color_map = {
                category: f'#{random.randint(0, 0xFFFFFF):06x}' 
                for category in unique_categories
            }


        colors_for_documents = [category_color_map[category] for category in categories]
        
        hover_texts = []
        for doc_id, meta in zip(doc_ids, metadatas):
             source = meta.get('source', 'Local') if meta else 'Unknown'
             filename = meta.get('filename', doc_id) if meta else doc_id
             hover_texts.append(f"ID: {doc_id}<br>Category: {meta.get('category', 'N/A') if meta else 'N/A'}<br>Source: {source}<br>File: {filename}")


        perplexity_value = min(30, embeddings.shape[0] - 1) 
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, n_iter=300)
        glog.info(f"Running t-SNE with perplexity={perplexity_value}...")
        reduced_embeddings = tsne.fit_transform(embeddings)
        glog.info("t-SNE calculation complete.")

        scatter = go.Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers',
            marker=dict(size=7, 
                        color=colors_for_documents,
                        opacity=0.8,
                        line=dict(width=0.5, color='DarkSlateGrey') 
                        ),
            text=hover_texts,
            hoverinfo='text' 
        )

        legend_entries = []
        for category, color in category_color_map.items():
            legend_entries.append(go.Scatter(
                x=[None], y=[None], 
                mode='markers',
                marker=dict(color=color, size=10),
                name=category, 
                showlegend=True
            ))

        layout = go.Layout(
            title="t-SNE Visualization of Document Embeddings by Category",
            xaxis=dict(title="t-SNE Dimension 1", showticklabels=False, zeroline=False),
            yaxis=dict(title="t-SNE Dimension 2", showticklabels=False, zeroline=False),
            showlegend=True,
            hovermode='closest',
            legend=dict(
                 title="Categories",
                 itemsizing='constant', 
                 traceorder='normal',
                 orientation='v', 
                 ), 
            margin=dict(l=20, r=200, t=50, b=20),
            width=1200,
            height=800,
        )

        fig = go.Figure(data=[scatter] + legend_entries, layout=layout) 

        return fig

    except Exception as e:
        glog.exception("Error generating embedding visualization:") 
        return f"An error occurred during visualization: {e}"


def add_document_interface(file: gr.File) -> str:
    """
    Allows the user to add a document by uploading a file. Processes various formats using MarkItDown.

    Args:
        file (gr.File): The file object representing the uploaded document.

    Returns:
        str: A message indicating the success or failure of the document upload and chunking.
    """
    if file is None:
        return "No file uploaded. Please select a document."

    file_path = file.name
    file_name = os.path.basename(file_path)
    glog.info(f"Adding document from uploaded file: {file_name}")

    try:
        md_converter = MarkItDown()
        result = md_converter.convert(file_path)
        content = result.text_content

        if not content.strip():
            return f"Warning: No text could be extracted from {file_name}."

        chunks = semantic_chunk_text(content, chunk_size=1000, overlap=200)
        glog.info(f"Splitting {file_name} into {len(chunks)} chunks.")

        added_count = 0
        skipped_count = 0
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                skipped_count += 1
                continue

            chunk_id = f"{file_name}_part{i}"
            abs_path = os.path.abspath(file_path)
            status = add_document(chunk_id, chunk, {
                "file_path": abs_path,
                "filename": file_name,
                "chunk_index": i,
                "category": "uploaded"
            })
            if "added successfully" in status:
                added_count += 1
            glog.debug(status)

        result_message = f"Document '{file_name}' processed. Added {added_count} new chunks."
        if skipped_count > 0:
            result_message += f" Skipped {skipped_count} empty chunks."

        return result_message

    except Exception as e:
        glog.error(f"Error processing uploaded file {file_name}: {e}")
        return f"Error processing file {file_name}: {e}"


def delete_document_interface(doc_id_prefix: str) -> str:
    """
    Allows the user to delete all chunks associated with a document prefix (e.g., filename).

    Args:
        doc_id_prefix (str): The prefix (like filename) of the document chunks to delete.

    Returns:
        str: A message indicating the result of the deletion.
    """
    if not doc_id_prefix:
        return "Please provide a document ID prefix (e.g., the filename) to delete."
        
    glog.info(f"Attempting to delete document chunks with prefix: {doc_id_prefix}")
    
    try:
        results = collection.get(where={"filename": doc_id_prefix}, include=[])
        ids_to_delete = results['ids']

        if not ids_to_delete:
            return f"No document chunks found with prefix or filename '{doc_id_prefix}'."

        glog.info(f"Found {len(ids_to_delete)} chunks to delete: {ids_to_delete}")
        collection.delete(ids=ids_to_delete) 
        
        return f"Successfully deleted {len(ids_to_delete)} chunks for document '{doc_id_prefix}'."

    except Exception as e:
        glog.error(f"Error deleting document chunks for '{doc_id_prefix}': {e}")
        return f"Error deleting document '{doc_id_prefix}': {e}"


initial_history = memory_manager.get_history()

with gr.Blocks(theme=gr.themes.Soft(font="ui-sans-serif")) as demo: 
    gr.Markdown('# RAGademic: Your University Study Assistant 🎓')

    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Row():
                 with gr.Column(scale=3):
                     chatbot_interface = gr.Chatbot(
                         value=initial_history, 
                         label="Conversation", 
                         elem_id="chatbot",
                         height=600,
                         show_copy_button=True,
                         bubble_full_width=False)
                     
                     user_input = gr.Textbox(
                         label="Enter your query:", 
                         placeholder="Ask about your notes, search arXiv, or just chat...",
                         lines=3)
                 
                 with gr.Column(scale=1):
                     model_selector = gr.Radio(
                         choices=["gpt-4o-mini", "llama3.2", "gemma3"],
                         value="gpt-4o-mini", 
                         label="Choose LLM")
                         
                     enable_local_rag_toggle = gr.Checkbox(
                         label="Use Local Knowledge Base",
                         value=True, 
                         elem_id="local_rag_toggle") 

                     enable_web_agent_toggle = gr.Checkbox(
                         label="Enable Web Search Agent (if Local KB is insufficient)", 
                         value=False, 
                         elem_id="web_agent_toggle")
                         
                     submit_button = gr.Button("Send Message", variant="primary")
                     
                     clear_memory_button = gr.Button("Clear Chat Memory")
                     
            def respond_stream(
                query: str,
                model: str,
                chat_hist: List[List[str]],
                enable_local_rag: bool,
                enable_web_agent: bool 
                ):
                """Handles the streaming response and updates history."""
                if not query.strip():
                    yield chat_hist
                    return

                glog.info(f"User query: '{query}', Model: {model}, Local RAG: {enable_local_rag}, Web Agent: {enable_web_agent}")

                summary = memory_manager.get_summary()

                chat_hist.append([query, None])
                yield chat_hist 

                response_generator = query_rag(
                    query,
                    model,
                    chat_hist[:-1], 
                    enable_local_rag, 
                    enable_web_agent, 
                    summary
                )

                full_response = ""
                ai_message_started = False
                sources_part = ""
                for chunk in response_generator:
                    if chunk:
                        if chunk.startswith("\n\n**Sources:**"):
                             sources_part = chunk
                             continue

                        if not ai_message_started and chunk.startswith(" "):
                           chunk = chunk.lstrip()

                        full_response += chunk
                        chat_hist[-1][1] = full_response
                        ai_message_started = True
                        yield chat_hist

                if sources_part:
                    if chat_hist[-1][1]:
                        chat_hist[-1][1] += sources_part
                    else:
                        chat_hist[-1][1] = sources_part
                    yield chat_hist

                glog.info(f"AI full response length: {len(full_response)}")

                final_ai_content = chat_hist[-1][1] if chat_hist[-1][1] else ""
                if final_ai_content and not final_ai_content.startswith("An error occurred"):
                    memory_manager.add_interaction(query, final_ai_content)
                    glog.info("Interaction added to long-term memory.")
                else:
                     glog.warning("Interaction not saved to memory due to empty or error response.")


            def clear_memory_ui():
                """Clears memory and updates the chatbot UI."""
                memory_manager.clear_memory()
                glog.info("Chatbot UI and memory cleared.")
                return [], "Chat memory cleared successfully."

            submit_button.click(
                respond_stream,
                inputs=[user_input, model_selector, chatbot_interface, enable_local_rag_toggle, enable_web_agent_toggle],
                outputs=[chatbot_interface]
            ).then(lambda: "", outputs=[user_input])

            user_input.submit(
                 respond_stream,
                inputs=[user_input, model_selector, chatbot_interface, enable_local_rag_toggle, enable_web_agent_toggle],
                outputs=[chatbot_interface]
            ).then(lambda: "", outputs=[user_input])

            clear_memory_status = gr.Textbox(label="Memory Status", interactive=False, visible=False) 
            clear_memory_button.click(
                clear_memory_ui, 
                inputs=[], 
                outputs=[chatbot_interface, clear_memory_status] 
            ).then(lambda: gr.update(visible=True), outputs=[clear_memory_status]) 

        with gr.TabItem("Knowledge Base Management"):
            gr.Markdown("Manage your documents and knowledge sources.")
            with gr.Accordion("Load Local Notes (PDFs)", open=False):
                 load_notes_button = gr.Button("Scan and Load Notes from 'knowledge_base' Folder")
                 load_notes_output = gr.Textbox(label="Loading Status", interactive=False, lines=3)
                 load_notes_button.click(fn=load_notes, outputs=[load_notes_output])

            with gr.Accordion("Add Single Document (PDF)", open=False):
                pdf_upload = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                add_doc_button = gr.Button("Add Uploaded Document")
                add_doc_output = gr.Textbox(label="Adding Status", interactive=False)
                add_doc_button.click(fn=add_document_interface, inputs=[pdf_upload], outputs=[add_doc_output])

            with gr.Accordion("Fetch Papers from arXiv", open=False):
                arxiv_search_box = gr.Textbox(label="Search Query (e.g., 'large language models')", placeholder="Enter topic...")
                arxiv_max_results = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Papers to Fetch")
                arxiv_search_button = gr.Button("Search arXiv and Add Abstracts")
                arxiv_search_output = gr.Textbox(label="Ingestion Status", interactive=False, lines=3)
                arxiv_search_button.click(fn=search_and_ingest_papers, inputs=[arxiv_search_box, arxiv_max_results], outputs=[arxiv_search_output])
                
            with gr.Accordion("Delete Document Chunks", open=False):
                 delete_doc_id_input = gr.Textbox(label="Document Filename or Prefix to Delete", placeholder="e.g., my_paper.pdf or arxiv_1234.5678")
                 delete_doc_button = gr.Button("Delete Document Chunks", variant="stop")
                 delete_doc_output = gr.Textbox(label="Deletion Status", interactive=False)
                 delete_doc_button.click(fn=delete_document_interface, inputs=[delete_doc_id_input], outputs=[delete_doc_output])

        with gr.TabItem("Embedding Visualization"):
            gr.Markdown("Visualize the relationships between your document chunks using t-SNE.")
            vis_button = gr.Button("Generate Visualization")
            vis_plot = gr.Plot()
            vis_status = gr.Textbox(label="Status", interactive=False, placeholder="Click button to generate plot...") 
            
            def generate_and_display_plot():
                """Handles plot generation and displays status/plot."""
                result = visualize_embeddings_interactive()
                if isinstance(result, str): 
                    return gr.update(value=None), result 
                else: 
                    return result, "Plot generated successfully."

            vis_button.click(fn=generate_and_display_plot, outputs=[vis_plot, vis_status])

if __name__ == "__main__":
    glog.setLevel("INFO")
    glog.info("Starting RAGademic Application...")
    vector_db_path = "./chroma_db" 
    if not os.path.exists(vector_db_path):
         os.makedirs(vector_db_path)
         glog.info(f"Created ChromaDB directory: {vector_db_path}")
    demo.launch()
    glog.info("RAGademic Application stopped.")