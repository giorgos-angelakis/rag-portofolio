import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load GROQ_API_KEY from .env
load_dotenv()

app = FastAPI(title="AI Document Chat API", version="1.0")

# 1. Load the FAISS vector database created in Phase 1
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 2. Initialize the Groq LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

# Helper function to combine retrieved chunks into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Build the prompt template
prompt_template = """You are an expert AI document assistant.
Use the retrieved context below to answer the user's question accurately.
If the context doesn't contain enough info, state that clearly.

Context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 4. Build the modern LCEL RAG chain
# - Converts the question into context via the retriever
# - Passes both context and question into the prompt -> LLM -> Output Parser
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define API Request Schema
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "online", "message": "RAG API is live!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Pass the input string directly to the chain
        answer = rag_chain.invoke(request.question)
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))