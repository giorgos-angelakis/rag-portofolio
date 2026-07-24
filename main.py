import os

# Lock ONNX and multi-threading to 1 thread BEFORE importing ML models
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    gc.collect()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server Safeguard: {str(exc)}"}
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
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # 1. Read metadata header
        reader = PdfReader(tmp_path)
        num_pages = len(reader.pages)
        
        # Set max page limit to 10 for free tier stability
        if num_pages > 10:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Document has {num_pages} pages. The free tier supports a maximum of 10 pages."
            )

        # 2. Extract plain text
        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={"page": i + 1}))

        os.remove(tmp_path)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        # 3. Smaller chunks (500 chars) = lower memory per matrix calculation
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        del documents
        gc.collect()

        # 4. MICRO-BATCH INGESTION (Prevents RAM Spikes):
        # Instead of embedding all chunks at once, process 2 chunks at a time
        vector_db = None
        batch_size = 2
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vector_db is None:
                vector_db = FAISS.from_documents(batch, embeddings)
            else:
                vector_db.add_documents(batch)
            gc.collect()  # Force memory release back to OS after every 2 chunks

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