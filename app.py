import glob
import os
import random
import urllib
import xml.etree.ElementTree as ET
from typing import List, Optional

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

from vector_db import (add_document, delete_document, list_documents,
                       query_documents)

load_dotenv(override=True)

_SYSTEM_PROMPT: str ="""
You are a helpful university tutor, students will ask you questions about their 
books, studies or articles. You may be provided with additional context for some questions.
If provided, answer based on the context. If you do not know something, clearly
state it.
"""
_BASE_PROMPT: str = """
Answer questions also based on the context.\n
"""
_MODEL_SIZE: str = os.getenv("MODEL_SIZE", "4b")
_KB_FOLDER: str = "knowledge_base/"

def load_notes() -> None:
    """
    Loads notes from the knowledge base folder and adds them to the vector DB.

    This function iterates through all categories and files in the knowledge
    base folder, extracts the text from PDF documents, chunks the content,
    and adds it to the vector database.

    Returns:
        None
    """
    for category in os.listdir(_KB_FOLDER):
        category_path = os.path.join(_KB_FOLDER, category)
        if os.path.isdir(category_path):
            for file_path in glob.glob(f"{category_path}/*.pdf"):
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    content = "\n".join([
                        page.extract_text() for page in reader.pages
                        if page.extract_text()
                    ])

                    chunks = semantic_chunk_text(content, chunk_size=2000, overlap=400)
                    doc_id = os.path.basename(file_path)

                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_part{i}"
                        add_document(
                            chunk_id, chunk, {
                                "category": category,
                                "file_path": os.path.abspath(file_path),
                                "chunk_index": i
                            })
                        
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
        separators=["\n\n", "\n", " ", ""]
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
    response = requests.get(url)

    if response.status_code != 200:
        return "Failed to fetch papers from arXiv."

    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    if not entries:
        return "No papers found."

    count = 0
    for entry in entries:
        title = entry.find("atom:title", ns).text.strip()
        abstract = entry.find("atom:summary", ns).text.strip()
        arxiv_id = entry.find("atom:id", ns).text.split('/')[-1]
        link = entry.find("atom:link[@type='text/html']", ns).attrib['href']

        full_text = f"Title: {title}\nAbstract: {abstract}\nLink: {link}"
        chunks = semantic_chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{arxiv_id}_part{i}"
            add_document(chunk_id, chunk, {
                "source": "arXiv",
                "category": f"arXiv_{query.lower().replace(' ', '_')}",
                "query": query,
                "arxiv_id": arxiv_id,
                "link": link,
                "chunk_index": i,
            })

        count += 1

    return f"Ingested {count} papers from arXiv for query '{query}'."

def query_rag(query: str, model: str, history: Optional[List[str]], use_rag: bool):
    if history is None:
        history = []

    context = ""
    sources = []
    if use_rag:
        results = query_documents(query)
        context = "\n\n".join([
            " ".join(doc) if isinstance(doc, list) else doc
            for sublist in results.get("documents", [])
            for doc in sublist
        ])

        used_docs = set()

        for i, (doc_group, meta_group) in enumerate(zip(results.get("documents", []), results.get("metadatas", []))):
            for doc_text, meta in zip(doc_group, meta_group):
                doc_id = meta.get("chunk_id") or meta.get("file_path")  
                if doc_id in used_docs:
                    continue
                used_docs.add(doc_id)

                title = meta.get("title") or meta.get("filename") or "Untitled"
                if len(title) > 60:
                    title = title[:57] + "..."

                preview = " ".join(doc_text.strip().split())[:120] + "..."
                file_path = meta.get("file_path")
                arxiv_id = meta.get("arxiv_id")
                chunk_index = meta.get("chunk_index", 0)

                url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"file://{urllib.parse.quote(file_path)}#page={chunk_index + 1}"

                sources.append(f"[{len(used_docs)}] [{title}]({url}) — {preview}")


    sources_text = "\n\n**Sources:**\n" + "\n".join(sources) if sources else "\n\n**Sources:**\n- No available sources."
    chat_history = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])
    prompt = _BASE_PROMPT + (f"Context:\n{context}\n\n" if use_rag else "")
    prompt += f"Chat History:\n{chat_history}\n\n Question: {query}\n"

    if model in ["llama3.2", "gemma3"]:
        if model == "gemma3":
            model = model + ":" + _MODEL_SIZE
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": chat_history + prompt}
            ],
            stream=True
        )
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
        yield sources_text

    else:
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
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        yield sources_text



def chatbot(query: str, model: str, history: List[List[str]],
            use_rag: bool) -> List[List[str]]:
    """
    Handles chatbot interaction by maintaining conversation history.

    Args:
        query (str): The user's query.
        model (str): The model to use (e.g., 'llama3.2' or 'gpt-4o-mini').
        history (List[List[str]]): The conversation history.
        use_rag (bool): Whether to retrieve relevant documents from the vector database.

    Returns:
        List[List[str]]: The updated conversation history.
    """
    if query.strip():
        response = query_rag(query, model, history, use_rag)
        history.append([query, response])
    return history


def visualize_embeddings_interactive() -> go.Figure:
    """
    Visualizes document embeddings interactively using Plotly and colors
    them based on their categories.

    Returns:
        go.Figure: Plotly figure object containing the interactive t-SNE plot.
    """
    documents = list_documents()
    glog.info(f"Number of documents: {len(documents['ids'])}")

    if not documents["ids"]:
        return "No documents available for visualization."

    embeddings = np.array(documents['embeddings'])
    doc_ids = documents['ids']
    categories = [
        metadata.get('category', 'Unknown')
        for metadata in documents['metadatas']
    ]

    if embeddings.shape[0] < 2:
        return "Not enough documents to generate visualization."

    perplexity = min(30, embeddings.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    random_colors = [
        f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(categories))
    ]
    category_color_map = {
        category: random_colors[i]
        for i, category in enumerate(categories)
    }

    colors_for_documents = [
        category_color_map[category] for category in categories
    ]

    scatter = go.Scatter(x=reduced_embeddings[:, 0],
                         y=reduced_embeddings[:, 1],
                         mode='markers',
                         marker=dict(size=5,
                                     color=colors_for_documents,
                                     opacity=0.9),
                         text=doc_ids,
                         hoverinfo='text')

    legend_entries = [
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(color=color, size=10),
                   name=category)
        for category, color in category_color_map.items()
    ]

    layout = go.Layout(
        title="t-SNE Visualization of Document Embeddings by Category",
        xaxis=dict(title="t-SNE Dimension 1"),
        yaxis=dict(title="t-SNE Dimension 2"),
        showlegend=True,
        hovermode='closest',
        legend=dict(x=1, y=1, traceorder='normal', orientation='v'),
        width=1200,
        height=800,
    )

    fig = go.Figure(data=[scatter] + legend_entries, layout=layout)

    return fig


def add_document_interface(file: gr.File) -> str:
    """
    Allows the user to add a document by uploading a PDF file.

    Args:
        file (gr.inputs.File): The file object representing the uploaded PDF.

    Returns:
        str: A message indicating the success of the document upload and chunking.
    """
    file_name = os.path.basename(file.name)
    reader = PdfReader(file)
    content = "\n".join(
        [page.extract_text() for page in reader.pages if page.extract_text()])

    # Chunking before ingestion
    chunks = semantic_chunk_text(content, chunk_size=1000, overlap=200)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_name}_part{i}"
        add_document(chunk_id, chunk, {
            "file_path": os.path.abspath(file.name),
            "chunk_index": i
        })

    return f"Document {file_name} added successfully in {len(chunks)} chunks."


def delete_document_interface(doc_id: str) -> str:
    """
    Allows the user to delete a document by ID.

    Args:
        doc_id (str): The unique document ID to be deleted.

    Returns:
        str: A message indicating the success of the document deletion.
    """
    delete_document(doc_id)
    return f"Document {doc_id} deleted successfully."

with gr.Blocks() as chat_interface:
    gr.Markdown('# RAGademic: query your notes!')
    chatbot_interface = gr.Chatbot(resizable=True)
    model_selector = gr.Radio(choices=["gpt-4o-mini", "llama3.2", "gemma3"],
                              value="gpt-4o-mini",
                              label="Choose Model")
    use_rag_toggle = gr.Checkbox(label="Enable RAG", value=True)
    user_input = gr.Textbox(label="Enter your query")

    def respond_stream(query: str, model: str, history: List[List[str]], use_rag: bool):
        response_gen = query_rag(query, model, history, use_rag)
        full_response = ""
        for chunk in response_gen:
            full_response += chunk
            yield "", history + [[query, full_response]]

    user_input.submit(
        respond_stream,
        [user_input, model_selector, chatbot_interface, use_rag_toggle],
        [user_input, chatbot_interface])

with gr.Blocks() as tab2:
    with gr.Tab("Fetch Papers from arXiv"):
        search_box = gr.Textbox(label="Search Query")
        search_button = gr.Button("Search and Add Papers")
        search_output = gr.Textbox(label="Output")

        search_button.click(fn=search_and_ingest_papers,
                            inputs=[search_box],
                            outputs=[search_output])
        
    with gr.Tab("Load Notes"):
        load_button = gr.Button("Load Notes")
        output = gr.Textbox(label="Output")
        load_button.click(fn=load_notes, outputs=output)

    with gr.Tab("Embedding Visualization"):
        gr.Interface(fn=visualize_embeddings_interactive,
                     inputs=[],
                     outputs=gr.Plot(),
                     title="Visualize Embeddings")

    with gr.Tab("Add Document"):
        gr.Interface(fn=add_document_interface,
                     inputs="file",
                     outputs="text",
                     title="Add Document")

    with gr.Tab("Delete Document"):
        gr.Interface(fn=delete_document_interface,
                     inputs="text",
                     outputs="text",
                     title="Delete Document")

tabs = gr.TabbedInterface([chat_interface, tab2],
                          tab_names=["Chat", "Database Management"])

if __name__ == "__main__":
    tabs.launch()
