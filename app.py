import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="AI Document Chat", page_icon="📄", layout="centered")

st.title("📄 AI Document Chat")
st.write("Upload any PDF document and chat with its contents in real-time.")

# Sidebar Configuration
st.sidebar.header("Backend Configuration")
# Input box for your backend URL (defaults to localhost for testing)
api_url = st.sidebar.text_input(
    "FastAPI Backend URL", 
    value="http://127.0.0.1:8000"
)

st.sidebar.markdown("---")
st.sidebar.header("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if st.sidebar.button("Process PDF", type="primary"):
    if uploaded_file is not None:
        with st.spinner("Ingesting and embedding document..."):
            try:
                # Package PDF binary stream to send over HTTP multipart/form-data
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{api_url}/upload", files=files)
                
                if response.status_code == 200:
                    st.sidebar.success(response.json().get("message", "Ingestion successful!"))
                else:
                    st.sidebar.error(f"Error {response.status_code}: {response.json().get('detail')}")
            except Exception as e:
                st.sidebar.error(f"Failed to connect to backend: {e}")
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
    # Render user query immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Query the FastAPI /ask endpoint via JSON
    with st.spinner("Searching context & generating response..."):
        try:
            payload = {"question": prompt}
            response = requests.post(f"{api_url}/ask", json=payload)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No response content.")
            else:
                answer = f"⚠️ Backend Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}"
        except Exception as e:
            answer = f"❌ Connection Error: Ensure your FastAPI server is running at `{api_url}`."

    # Render assistant output
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)