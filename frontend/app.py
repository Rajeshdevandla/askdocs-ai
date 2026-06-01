import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AskDocs AI", page_icon="📄", layout="centered")
st.title("📄 AskDocs AI")
st.caption("Upload a PDF and ask questions about it")

# --- sidebar: upload section ---
with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and "session_id" not in st.session_state:
        with st.spinner("Processing PDF..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.document_name = data["document_name"]
                    st.session_state.messages = []
                    st.success(
                        f"Ready! {data['page_count']} pages, {data['chunk_count']} chunks indexed."
                    )
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the server running?")

    if "session_id" in st.session_state:
        st.info(f"Document: {st.session_state.document_name}")
        if st.button("Upload new document"):
            for key in ["session_id", "document_name", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

# --- main chat area ---
if "session_id" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# show chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.write(f"Page {c['page']} (relevance: {c['score']})")

# chat input
question = st.chat_input("Ask something about your document...")

if question:
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"session_id": st.session_state.session_id, "question": question},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])
                    if data.get("citations"):
                        with st.expander("Sources"):
                            for c in data["citations"]:
                                st.write(f"Page {c['page']} (relevance: {c['score']})")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "citations": data.get("citations", []),
                        }
                    )
                else:
                    error_msg = response.json().get("detail", "Something went wrong")
                    st.error(error_msg)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the server running?")
