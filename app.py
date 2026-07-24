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
    value="https://rag-ai-document-chat.onrender.com"
)

st.sidebar.markdown("---")
st.sidebar.header("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if st.sidebar.button("Process PDF", type="primary"):
    if uploaded_file is not None:
        with st.spinner("Ingesting and embedding document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{api_url}/upload", files=files)
                
                # Check if the server returned JSON or an HTML error page
                try:
                    res_data = response.json()
                except Exception:
                    res_data = None

                if response.status_code == 200 and res_data:
                    st.sidebar.success(res_data.get("message", "Ingestion successful!"))
                elif res_data and "detail" in res_data:
                    st.sidebar.error(f"⚠️ {res_data['detail']}")
                else:
                    st.sidebar.error(f"❌ Server Error ({response.status_code}): The server ran out of memory or crashed.")
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
    # Add user query to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Query the FastAPI /ask endpoint via JSON
    with st.spinner("Searching context & generating response..."):
        try:
            payload = {"question": prompt}
            response = requests.post(f"{api_url}/ask", json=payload)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No response content.")
            elif response.status_code == 400:
                # Capture the 400 error message directly from FastAPI
                error_detail = response.json().get("detail", "Please upload a PDF first.")
                answer = f"⚠️ **Cannot process request:** {error_detail}"
            else:
                answer = f"⚠️ **Backend Error ({response.status_code}):** {response.json().get('detail', 'Unknown error')}"
        except Exception as e:
            answer = f"❌ **Connection Error:** Could not reach the server at `{api_url}`."

    # Render assistant output
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)