from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_document(pdf_path: str, vector_db_path: str, max_pages: int = 100):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Safeguard: Reject files exceeding capacity before running heavy processing
    page_count = len(documents)
    if page_count > max_pages:
        raise ValueError(f"Document has {page_count} pages, which exceeds the max limit of {max_pages} pages.")

    print(f"Splitting {page_count} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Loading FastEmbed model with batched execution (ONNX Runtime)...")
    # batch_size=64 optimizes thread execution and prevents CPU saturation on free hosting
    embeddings = FastEmbedEmbeddings(batch_size=64)

    print(f"Creating vector database for {len(chunks)} chunks...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(vector_db_path)
    print("Done! Database re-saved with FastEmbed.")

if __name__ == "__main__":
    ingest_document("sample_paper.pdf", "faiss_index")
