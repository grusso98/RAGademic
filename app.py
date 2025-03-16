import os
import random
import gradio as gr
import glob
import numpy as np
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import urllib
from vector_db import (add_document, query_documents, delete_document,
                       list_documents)
from PyPDF2 import PdfReader
from openai import OpenAI
from typing import Generator, List, Optional
import glog
import plotly.graph_objs as go


def chunk_text(text: str,
               chunk_size: int = 1000,
               overlap: int = 200) -> List[str]:
    """
    Splits text into overlapping chunks of the specified size.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk.
            Defaults to 1000.
        overlap (int, optional): The number of overlapping characters between
            consecutive chunks. Defaults to 200.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def load_notes() -> None:
    """
    Loads notes from the knowledge base folder and adds them to the vector DB.

    This function iterates through all categories and files in the knowledge
    base folder, extracts the text from PDF documents, chunks the content,
    and adds it to the vector database.

    Returns:
        None
    """
    for category in os.listdir(KB_FOLDER):
        category_path = os.path.join(KB_FOLDER, category)
        if os.path.isdir(category_path):
            for file_path in glob.glob(f"{category_path}/*.pdf"):
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    content = "\n".join([
                        page.extract_text() for page in reader.pages
                        if page.extract_text()
                    ])

                    chunks = chunk_text(content, chunk_size=1000, overlap=200)
                    doc_id = os.path.basename(file_path)

                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_part{i}"
                        add_document(
                            chunk_id, chunk, {
                                "category": category,
                                "file_path": os.path.abspath(file_path),
                                "chunk_index": i
                            })


def query_rag(
        query: str,
        model: str,
        history: Optional[List[str]] = None) -> Generator[str, None, None]:
    """
    Handles querying the vector DB and generating a streaming response.

    Args:
        query (str): The query text to be asked.
        model (str): The model to use (e.g., 'llama3.2' or 'gpt-4o-mini').
        history (Optional[List[str]], optional): Previous chat history.

    Yields:
        str: A progressively generated response from the model.
    """
    if history is None:
        history = []

    results = query_documents(query)
    context = "\n\n".join([
        " ".join(doc) if isinstance(doc, list) else doc
        for sublist in results.get("documents", [])  # Iterate over sublists
        for doc in sublist  # Extract each document inside the sublists
    ])

    sources = []
    for i, meta in enumerate(results.get("metadatas", [])):
        doc_text = results["documents"][i][0]

        if isinstance(meta, list):
            meta = meta[0]
            if isinstance(meta, dict):
                file_path = meta.get("file_path", "#")
                clean_text = " ".join(
                    doc_text.split())  # Remove excessive spaces/newlines
                chunk_text = clean_text[:200].rsplit(
                    " ", 1)[0] + "..." if len(clean_text) > 200 else clean_text
                chunk_index = meta.get("chunk_index", 0)
            else:
                file_path = "#"
                chunk_text = "No preview available"
                chunk_index = 0

        if file_path and file_path != "#":
            encoded_path = urllib.parse.quote(file_path)
            pdf_viewer_url = f"file://{encoded_path}#page={chunk_index + 1}"
            sources.append(
                f'- <a href="{pdf_viewer_url}" target="_blank">{chunk_text.replace("\n", " ")[:150]}...</a>'
            )

        else:
            sources.append(f"- {chunk_text}...")

    sources_text = "\n\n**Sources:**\n" + "\n".join(
        sources) if sources else "\n\n**Sources:**\n- No available sources."

    prompt = f"You are a helpful university tutor. Answer the question based on the provided notes. Also state the sources. \n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    if model == "llama3.2":
        openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    elif model == "gpt-4o-mini":
        openai = OpenAI()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
        if OPENAI_API_BASE:
            openai.api_base = OPENAI_API_BASE

    response = openai.chat.completions.create(model=model,
                                              messages=[{
                                                  "role":
                                                  "system",
                                                  "content":
                                                  "You are a university tutor."
                                              }, {
                                                  "role": "user",
                                                  "content": prompt
                                              }],
                                              stream=True)

    full_response = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[
                0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield full_response

    yield full_response + sources_text


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
    chunks = chunk_text(content, chunk_size=1000, overlap=200)

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


def chatbot(query, model, history):
    """
    Handles the chatbot interaction by maintaining conversation history.

    Args:
        query (str): The user's query.
        model (str): The model to use (e.g., 'llama3.2' or 'gpt-4o-mini').
        history (List[Tuple[str, str]]): The conversation history.

    Returns:
        List[Tuple[str, str]]: The updated conversation history.
    """
    # Append the user message to the history if it's not empty
    if query.strip():
        history.append((query, ""))

    # Generate the assistant's response
    response_text = list(
        query_rag(query, model, [msg[0] for msg in history if msg[1] == ""]))
    response = response_text[-1] if response_text else ""

    # Append the assistant's response to the history if it's not empty
    if response.strip():
        history.append(("", response))

    return history


# Load OpenAI API key
load_dotenv()

# Knowledge base folder
KB_FOLDER = "knowledge_base"

# Load notes at startup
load_notes()

# Gradio interface for chat-like interaction
with gr.Blocks() as chat_interface:
    chatbot_interface = gr.Chatbot()
    model_selector = gr.Radio(choices=["gpt-4o-mini", "llama3.2"],
                              value="gpt-4o-mini",
                              label="Choose Model")
    user_input = gr.Textbox(label="Enter your query")

    def respond(query, model, history):
        updated_history = chatbot(query, model, history)
        return gr.update(value=""), updated_history

    user_input.submit(respond, [user_input, model_selector, chatbot_interface],
                      [user_input, chatbot_interface])

tab2 = gr.TabbedInterface(
    [
        gr.Interface(fn=visualize_embeddings_interactive,
                     inputs=[],
                     outputs=gr.Plot(),
                     title="Visualize Embeddings"),
        gr.Interface(fn=add_document_interface,
                     inputs="file",
                     outputs="text",
                     title="Add Document"),
        gr.Interface(fn=delete_document_interface,
                     inputs="text",
                     outputs="text",
                     title="Delete Document")
    ],
    tab_names=["Embedding Visualization", "Add Document", "Delete Document"])

tabs = gr.TabbedInterface([chat_interface, tab2],
                          tab_names=["Chat", "Database Management"])

if __name__ == "__main__":
    tabs.launch()
