import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="AI Document Chat", page_icon="📄", layout="centered")

st.title("📄 AI Document Chat")
st.write("Upload any PDF document and chat with its contents in real-time.")

# Sidebar Configuration
st.sidebar.header("Backend Configuration")
api_url = st.sidebar.text_input(
    "FastAPI Backend URL", 
    value="https://rag-ai-document-chat.onrender.com"
)

st.sidebar.markdown("---")
st.sidebar.header("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
st.sidebar.caption("⚠️ **Limit:** 100 pages max or 20 MB (Free Tier)")

MAX_FILE_SIZE_MB = 20

if st.sidebar.button("Process PDF", type="primary"):
    if uploaded_file is not None:
        # Pre-flight check: Prevent uploading massive files to save backend bandwidth
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.sidebar.error(f"⚠️ File size ({file_size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit.")
        else:
            with st.spinner("Ingesting and embedding document... (This may take up to 1-2 minutes for larger PDFs)"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    # Increased timeout to 180s to accommodate processing 50-100 page files on CPU
                    response = requests.post(f"{api_url}/upload", files=files, timeout=180)
                    
                    try:
                        res_data = response.json()
                    except Exception:
                        res_data = {
                            "detail": f"Server non-JSON response (HTTP {response.status_code}). The backend server timed out or ran out of RAM."
                        }

                    if response.status_code == 200:
                        st.sidebar.success(res_data.get("message", "Ingestion successful!"))
                    else:
                        error_msg = res_data.get("detail", "An unknown error occurred.")
                        st.sidebar.error(f"⚠️ {error_msg}")

                except requests.exceptions.Timeout:
                    st.sidebar.error("❌ Request timed out: The backend is taking longer than 3 minutes to process the file.")
                except Exception as e:
                    st.sidebar.error(f"❌ Connection Error: Could not connect to backend (`{e}`)")
    else:
        st.sidebar.warning("Please select a PDF file first.")

# Chat Session State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Searching context & generating response..."):
        try:
            payload = {"question": prompt}
            response = requests.post(f"{api_url}/ask", json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No response content.")
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Please upload a PDF first.")
                answer = f"⚠️ **Cannot process request:** {error_detail}"
            else:
                answer = f"⚠️ **Backend Error ({response.status_code}):** {response.json().get('detail', 'Unknown error')}"
        except requests.exceptions.Timeout:
            answer = "❌ **Timeout Error:** Query took too long to respond."
        except Exception as e:
            answer = f"❌ **Connection Error:** Could not reach the server at `{api_url}`."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
