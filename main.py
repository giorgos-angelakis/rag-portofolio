import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="AI Document Chat API", version="2.2")

# Global variables for state
embeddings = FastEmbedEmbeddings()
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
        # Save incoming file to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load and chunk the uploaded PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Re-index chunks using FastEmbed and FAISS
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # Rebuild LCEL RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Delete temp file
        os.remove(tmp_path)

        return {"status": "success", "message": f"Successfully ingested '{file.filename}'!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # If no PDF has been uploaded via /upload, block the query
    if rag_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="⚠️ No document has been uploaded yet. Please upload a PDF in the sidebar first!"
        )
    try:
        answer = rag_chain.invoke(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))