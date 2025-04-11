import glob
import os
import random
import urllib
import xml.etree.ElementTree as ET
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
import glog
import gradio as gr
import numpy as np
import ollama
import plotly.graph_objs as go
import requests
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.manifold import TSNE

from memory_manager import MemoryManager 

from vector_db import (add_document, list_documents,
                       query_documents, collection)

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

Context from Documents (if available):
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

def load_notes() -> None:
    """
    Loads notes from the knowledge base folder and adds them to the vector DB.

    This function iterates through all categories and files in the knowledge
    base folder, extracts the text from PDF documents, chunks the content,
    and adds it to the vector database.

    Returns:
        None
    """
    loaded_count = 0
    for category in os.listdir(_KB_FOLDER):
        category_path = os.path.join(_KB_FOLDER, category)
        if os.path.isdir(category_path):
            for file_path in glob.glob(f"{category_path}/*.pdf"):
                glog.info(f"Processing file: {file_path}")
                try:
                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        content = "\n".join([
                            page.extract_text() for page in reader.pages
                            if page.extract_text()
                        ])

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
                                    "filename": doc_id_base # Add filename for display
                                })
                        loaded_count += 1
                        glog.info(f"Added {len(chunks)} chunks for {doc_id_base}")
                except Exception as e:
                    glog.error(f"Failed to process {file_path}: {e}")
    return f"Finished loading notes. Processed {loaded_count} files." # Return status

def semantic_chunk_text(text: str, chunk_size: int = 2000, overlap: int = 400) -> List[str]:
    """
    Splits a given text into semantic chunks using a recursive character text splitter.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        overlap (int, optional): The amount of overlap between consecutive chunks. Defaults to 200.

    Returns:
        List[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len # Ensure length_function is specified
    )
    return splitter.split_text(text)


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

def query_rag(query: str, model: str, history: List[List[str]], use_rag: bool, memory_summary: str):
    """
    Queries the RAG system, incorporating long-term memory summary.

    Args:
        query (str): The user's query.
        model (str): The model identifier (e.g., 'llama3.2:latest', 'gpt-4o-mini').
        history (List[List[str]]): The *current session's* chat history.
        use_rag (bool): Whether to use RAG context.
        memory_summary (str): The summary of past conversations.

    Yields:
        str: Chunks of the generated response.
    """
    context = ""
    sources = []
    if use_rag:
        glog.info(f"RAG enabled. Querying vector DB for: '{query}'")
        try:
            results = query_documents(query, n_results=5) 
            if results and results.get("documents"):
                context_docs = []
                for doc_list in results.get("documents", []):
                     if isinstance(doc_list, list): 
                         context_docs.extend(doc_list)
                     elif isinstance(doc_list, str):
                         context_docs.append(doc_list)

                context = "\n\n".join(context_docs)
                glog.info(f"Retrieved {len(context_docs)} context snippets.")

                used_docs_info = {} 

                # Iterate through metadata which should align with documents
                for meta_list in results.get("metadatas", []):
                     if not meta_list: continue
                     for meta in meta_list:
                        if not meta: continue 

                        # Determine a unique identifier for the source document/chunk
                        file_path = meta.get("file_path")
                        arxiv_id = meta.get("arxiv_id")
                        chunk_index = meta.get("chunk_index", 0)
                        
                        # Prefer file_path or arxiv_id as base identifier
                        doc_base_id = file_path if file_path else (f"arxiv_{arxiv_id}" if arxiv_id else None)
                        if not doc_base_id: continue # Cannot identify source

                        # Create a more unique key including chunk index if needed, or just use base ID
                        source_key = f"{doc_base_id}_chunk{chunk_index}" # Unique key per chunk

                        if source_key not in used_docs_info:
                            title = meta.get("title") or meta.get("filename") or doc_base_id.split('/')[-1] or "Unknown Source"
                            if len(title) > 60:
                                title = title[:57] + "..."

                            preview = context[:120].strip().replace('\n', ' ') + "..." # Generic preview for now


                            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (f"file://{urllib.parse.quote(file_path)}" if file_path else "#")
                            
                            # Adjust URL for page number if chunk index is meaningful (e.g., for PDFs)
                            if file_path and isinstance(chunk_index, int) and chunk_index > 0:
                                # Rough estimation, assumes chunks correspond somewhat to pages
                                url += f"#page={chunk_index + 1}" 
                                
                            used_docs_info[source_key] = {
                                "title": title,
                                "url": url,
                                "preview": preview 
                            }
                
                source_links = []
                for i, info in enumerate(used_docs_info.values()):
                     source_links.append(f"[{i+1}] [{info['title']}]({info['url']})") # Removed preview for cleaner look
                
                sources_text = "\n\n**Sources:**\n" + "\n".join(source_links) if source_links else "\n\n**Sources:**\n- Context retrieved but source mapping failed."


            else:
                glog.info("No relevant documents found in vector DB.")
                context = "No relevant documents found."
                sources_text = "\n\n**Sources:**\n- No relevant documents found."
        except Exception as e:
            glog.error(f"Error querying vector DB: {e}")
            context = "Error retrieving context from documents."
            sources_text = "\n\n**Sources:**\n- Error retrieving sources."
    else:
        glog.info("RAG is disabled.")
        context = "RAG Disabled" # Indicate RAG was off
        sources_text = "\n\n**Sources:**\n- RAG Disabled."

    history_limit = 5 # Show last N turns in prompt
    recent_history = history[-history_limit:]
    chat_history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in recent_history]) if recent_history else "No current session history."

    prompt = _BASE_PROMPT.format(
        memory_summary=memory_summary if memory_summary else "No long-term memory summary available.",
        context=context,
        chat_history=chat_history_text,
        query=query
    )
    

    try:
        if model in ["llama3.2", "gemma3"]:
            llm_model_id = model
            if model == "gemma3":
                llm_model_id = f"{model}:{_MODEL_SIZE}" 
            elif model == "llama3.2":
                 # Assuming a default tag like 'latest' if not specified
                 # Check if ollama needs explicit tag, e.g., 'llama3.2:latest'
                 if ':' not in llm_model_id:
                      llm_model_id += ":latest" # Adjust if needed

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
            yield sources_text 

        elif "gpt" in model: 
            glog.info(f"Streaming response from OpenAI model: {model}")
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("OPENAI_API_BASE"):
                openai_client.api_base = os.getenv("OPENAI_API_BASE")
            
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
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    full_response_content += content_piece
                    yield content_piece 
            yield sources_text 
        else:
             glog.error(f"Model '{model}' not supported.")
             yield f"Error: Model '{model}' is not configured or supported."
             yield sources_text 

    except Exception as e:
        glog.error(f"Error during LLM call with model {model}: {e}")
        yield f"An error occurred while generating the response: {e}"
        yield sources_text 

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
    Allows the user to add a document by uploading a PDF file. Chunks the document.

    Args:
        file (gr.File): The file object representing the uploaded PDF. Can be None.

    Returns:
        str: A message indicating the success or failure of the document upload and chunking.
    """
    if file is None:
        return "No file uploaded. Please select a PDF file."
        
    file_path = file.name 
    file_name = os.path.basename(file_path)
    
    if not file_name.lower().endswith(".pdf"):
         return f"Error: Invalid file type '{os.path.splitext(file_name)[1]}'. Please upload a PDF file."

    glog.info(f"Adding document from uploaded file: {file_name}")
    
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            content = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )

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
                         
                     use_rag_toggle = gr.Checkbox(
                         label="Enable RAG (Use My Notes)", 
                         value=True)
                         
                     submit_button = gr.Button("Send Message", variant="primary")
                     
                     clear_memory_button = gr.Button("Clear Chat Memory")
                     
            def respond_stream(query: str, model: str, chat_hist: List[List[str]], use_rag: bool):
                """Handles the streaming response and updates history."""
                if not query.strip():
                    yield chat_hist 
                    return

                glog.info(f"User query: '{query}', Model: {model}, RAG: {use_rag}")
                
                summary = memory_manager.get_summary() 
                
                chat_hist.append([query, None]) 
                yield chat_hist

                response_generator = query_rag(query, model, chat_hist[:-1], use_rag, summary) 

                full_response = ""
                for chunk in response_generator:
                    if chunk: 
                        if full_response == "" and chunk.startswith(" "): 
                           chunk = chunk.lstrip()
                        full_response += chunk
                        chat_hist[-1][1] = full_response 
                        yield chat_hist
                
                glog.info(f"AI full response length: {len(full_response)}")
                
                if full_response and not full_response.startswith("An error occurred"):
                    memory_manager.add_interaction(query, full_response)
                    glog.info("Interaction added to long-term memory.")
                else:
                     glog.warning("Interaction not saved to memory due to empty or error response.")


            def clear_memory_ui():
                """Clears memory and updates the chatbot UI."""
                memory_manager.clear_memory()
                glog.info("Chatbot UI cleared.")
                return [], "Chat memory cleared successfully." 

            submit_button.click(
                respond_stream,
                inputs=[user_input, model_selector, chatbot_interface, use_rag_toggle],
                outputs=[chatbot_interface]
            ).then(lambda: "", outputs=[user_input]) 

            user_input.submit( 
                 respond_stream,
                inputs=[user_input, model_selector, chatbot_interface, use_rag_toggle],
                outputs=[chatbot_interface]
            ).then(lambda: "", outputs=[user_input]) 

            clear_memory_status = gr.Textbox(label="Memory Status", interactive=False, visible=False) # Hidden status box
            clear_memory_button.click(
                clear_memory_ui, 
                inputs=[], 
                outputs=[chatbot_interface, clear_memory_status] # Clear chat UI and show status
            ).then(lambda: gr.update(visible=True), outputs=[clear_memory_status]) # Make status visible after click


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
    demo.launch()
    glog.info("RAGademic Application stopped.")