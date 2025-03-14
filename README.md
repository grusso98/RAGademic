# RAGademic: Query Your University Notes

This project provides a **Retrieval-Augmented Generation (RAG)** system to query your university notes efficiently (could be used to retrieve any kind of document really, but this is the use case i've opted for! :) ). It allows users to upload PDF documents, store them in a **Chroma** vector database, and interact with them through an LLM-powered chatbot. The project also includes a visualization tool for document embeddings.

## Features
- **Chat with your notes**: Ask questions, and the system retrieves relevant information using embeddings.
- **Database management**: Upload, delete, and visualize document embeddings.
- **Two LLM options**: Use either OpenAI's API or a local **LLaMA** model via an OpenAI-compatible API.
- **Two Embedding options**: Use either OpenAI's embedder or a HF embedder from the hub.

---

## Installation

### Prerequisites
Ensure you have Python **3.8+** installed.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ragademic.git
cd ragademic
```
### 2. Create Env end install requirements
```bash
python3 -m venv ragademic
pip install -r requirements.txt
```
Also create .env file and add:
```bash
OPENAI_API_KEY=<your_key>
EMBEDDER_TYPE=huggingface
HF_TOKEN=<your_token> 
```
Notes: 
- you can use llama3.2 installing it locally with ollama
- the openai api key is needed if you use gpt for chatting and/or openai embeddings
- embedder_type can either be **huggingface** or **openai**
- the huggingface embedding is obtained through API but it could be done locally too.

### 3. Load your PDF
Load PDFs 
```bash
./knowledge_base
    /category_1/*.pdf
    ...
    /category_n/*.pdf
```

### 4. Run the app
```bash
python3 app.py
```
#### Chat Interface
![Chat Interface](./imgs/chat.png)
#### DB Management Interface
![DB management Interface](./imgs/db_management.png)