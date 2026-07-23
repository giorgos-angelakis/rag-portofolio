# AI Document Chat — Production-Ready RAG API

A high-performance, lightweight **Retrieval-Augmented Generation (RAG)** API built with Python, FastAPI, LangChain (LCEL), FastEmbed, FAISS, and Groq. 

This service allows users to ingest PDF documents, convert text into vector embeddings, and perform intelligent, context-aware querying using Llama 3.3 70B.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange.svg)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

🚀 **[Click Here to Test the Live Interactive API Demo](https://your-app-name.onrender.com/docs)**

---

## 🌐 Live Interactive Demo

Try the API directly in your browser without cloning or installing anything:

👉 **[https://rag-ai-document-chat.onrender.com/docs](https://rag-ai-document-chat.onrender.com/docs)**

> *Note: Hosted on Render's free tier. If the service has been idle, the initial request may take ~30 seconds to wake up the server.*

---

## 🌟 Key Features

* **Memory-Optimized Embeddings**: Uses `FastEmbed` (ONNX Runtime) instead of heavy PyTorch runtimes, keeping total memory consumption under **150MB RAM** for free-tier cloud deployment.
* **Ultra-Fast Inferences**: Powered by Groq's high-speed Llama 3.3 70B inference engine.
* **Modern LCEL Pipeline**: Utilizes LangChain Expression Language (LCEL) for stable, future-proof RAG chains.
* **Vector Search**: Local vector similarity search backed by **FAISS**.
* **Containerized & Cloud Ready**: Fully packaged with Docker and pre-configured for instant deployment on cloud hosts like Render or Cloud Run.
* **Interactive API Documentation**: Auto-generated OpenAPI/Swagger UI endpoints available out of the box.

---

## 🛠️ Architecture & Workflow

```
[ PDF Document ] ──> PyPDFLoader ──> Text Splitter ──> FastEmbed (ONNX) ──> FAISS Vector Index
                                                                                 │
[ User Question ] ──> FastAPI (/ask) ──> Similarity Retriever <──────────────────┘
                                              │
                                              ▼
                             Context + Prompt Template ──> Groq (Llama 3.3 70B) ──> Structured Answer
```

---

## 🚀 Tech Stack

* **Language**: Python 3.10
* **API Framework**: FastAPI + Uvicorn
* **RAG Framework**: LangChain (`langchain-core`, `langchain-community`, `langchain-groq`)
* **Embedding Model**: FastEmbed (`BAAI/bge-small-en-v1.5` via ONNX Runtime)
* **Vector Store**: FAISS
* **LLM Engine**: Groq API (`llama-3.3-70b-versatile`)
* **Deployment**: Docker & Render

---

## 🏁 Quickstart (Local Development)

### 1. Prerequisites
* Python 3.10 or higher
* A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/rag-portfolio.git
cd rag-portfolio

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5. Ingest a Document
Place your PDF in the root directory (e.g., `sample_paper.pdf`) and run the ingestion script:
```bash
python ingest.py
```
This generates a local `faiss_index/` vector database directory.

### 6. Start the API Server
```bash
uvicorn main:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` in your browser to interact with the API via Swagger UI.

---

## 🐳 Running with Docker

### Build Image
```bash
docker build -t rag-document-chat .
```

### Run Container
```bash
docker run -p 8000:8000 -e GROQ_API_KEY="your_groq_api_key_here" rag-document-chat
```
Access the service at `http://localhost:8000/docs`.

---

## 📡 API Reference

### `GET /`
Health check endpoint to confirm the service is operational.

**Response:**
```json
{
  "status": "online",
  "message": "RAG API is live!"
}
```

### `POST /ask`
Submit a question to query the ingested document context.

**Request Body:**
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "question": "What is the main topic of the document?",
  "answer": "The document primarily discusses..."
}
```

---

## ☁️ Deployment on Render

This repository includes a `Dockerfile` optimized for low-memory container environments.

1. Connect your repository to **Render**.
2. Create a new **Web Service** selecting **Docker** as the runtime environment.
3. Set the instance type to **Free**.
4. Add `GROQ_API_KEY` under **Environment Variables**.
5. Deploy!
