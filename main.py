import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="AI Document Chat API", version="1.0")

# 1. Load lightweight FastEmbed embeddings (<120MB total RAM)
embeddings = FastEmbedEmbeddings()

# 2. Load FAISS index
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize Groq LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = """You are an expert AI document assistant.
Use the retrieved context below to answer the user's question accurately.
If the context doesn't contain enough info, state that clearly.

Context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "online", "message": "RAG API is live!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = rag_chain.invoke(request.question)
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))