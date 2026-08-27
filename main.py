import os

# Lock threading before importing ML libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import gc
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize global model objects
embeddings = FastEmbedEmbeddings(threads=1, batch_size=16)
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2)

# Lifespan manager: Warm up the ONNX model at boot time to prevent in-request RAM spikes
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load ONNX weights into RAM before receiving user requests
    embeddings.embed_query("warmup query")
    gc.collect()
    yield

app = FastAPI(title="AI Document Chat API", version="2.0", lifespan=lifespan)

vector_db = None
retriever = None
rag_chain = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = """You are an expert AI document assistant.
Use the retrieved context below to answer the user's question accurately.
If the context doesn't contain enough info, state that clearly.

Context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(prompt_template)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "online", "message": "RAG API is live!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db, retriever, rag_chain

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Write PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Parse PDF metadata and pages
        reader = PdfReader(tmp_path)
        num_pages = len(reader.pages)

        MAX_PAGES = 100
        if num_pages > MAX_PAGES:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Document has {num_pages} pages. The backend limit is set to {MAX_PAGES} pages."
            )

        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={"page": i + 1}))

        # Delete temp file and reader object immediately to free memory
        del reader
        os.remove(tmp_path)
        gc.collect()

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        # Text chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        del documents
        gc.collect()

        # Batch ingestion with smaller micro-batches (16 chunks) to keep RAM bounded on 512MB free tier
        vector_db = None
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vector_db is None:
                vector_db = FAISS.from_documents(batch, embeddings)
            else:
                vector_db.add_documents(batch)
            gc.collect()

        del chunks
        gc.collect()

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return {"status": "success", "message": f"Successfully ingested '{file.filename}' ({num_pages} pages)!"}

    except HTTPException as he:
        raise he
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="⚠️ No document uploaded. Please upload a PDF in the sidebar first."
        )
    try:
        answer = rag_chain.invoke(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
