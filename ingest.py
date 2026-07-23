from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_document(pdf_path: str, vector_db_path: str):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(vector_db_path)
    print("Done! Database saved.")

if __name__ == "__main__":
    ingest_document("sample_paper.pdf", "faiss_index")