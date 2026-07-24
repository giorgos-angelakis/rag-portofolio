import os

# Lock ONNX single-threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import tempfile
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

app = FastAPI(title="AI Document Chat API", version="2.0")

# SAFEGUARD 1: Global Catch-All Handler for any unhandled Python exception
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    gc.collect()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server Safeguard: An unexpected error occurred: {str(exc)}"}
    )

embeddings = FastEmbedEmbeddings(threads=1)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

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
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        num_pages = len(reader.pages)
        
        if num_pages > 15:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Document has {num_pages} pages. Free tier supports a max of 15 pages."
            )

        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={"page": i + 1}))

        os.remove(tmp_path)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        del documents
        gc.collect()

        # Build FAISS index
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        gc.collect()

        return {"status": "success", "message": f"Successfully ingested '{file.filename}' ({num_pages} pages)!"}

    # SAFEGUARD 2: Explicitly catch Python Memory Errors
    except MemoryError:
        gc.collect()
        raise HTTPException(
            status_code=400, 
            detail="Memory Limit Reached: This document requires too much memory to process on the free tier."
        )
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