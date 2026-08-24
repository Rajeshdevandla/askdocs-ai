import hashlib

import streamlit as st

from core.rag_pipeline import RAGPipeline
from frontend.deployment import load_uploaded_pdfs

st.set_page_config(page_title="AskDocs AI", page_icon="📄", layout="centered")
st.title("📄 AskDocs AI")
st.caption("Upload a PDF and ask grounded questions — no API key required in demo mode.")

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

with st.sidebar:
    st.header("Upload Documents")
    st.caption("Public demo: answers are deterministic and no document is sent to an external LLM.")
    uploaded_files = st.file_uploader(
        "Choose one or more text-based PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_upload_{st.session_state.uploader_version}",
    )

    if uploaded_files:
        uploads = [(item.name, item.getvalue()) for item in uploaded_files]
        document_hash = hashlib.sha256(
            b"".join(
                name.encode("utf-8") + b"\0" + contents
                for name, contents in uploads
            )
        ).hexdigest()

        if st.session_state.get("document_hash") != document_hash:
            with st.spinner("Extracting and indexing the PDF..."):
                try:
                    pipeline, documents = load_uploaded_pdfs(
                        uploads,
                        RAGPipeline,
                    )
                except Exception as exc:
                    st.error(f"Could not process this PDF: {exc}")
                else:
                    st.session_state.pipeline = pipeline
                    st.session_state.document_hash = document_hash
                    st.session_state.document_names = [
                        document["document_name"] for document in documents
                    ]
                    st.session_state.messages = []
                    total_pages = sum(document["page_count"] for document in documents)
                    total_chunks = sum(document["chunk_count"] for document in documents)
                    st.success(
                        f"Ready! {len(documents)} document(s), {total_pages} pages, "
                        f"and {total_chunks} chunks indexed."
                    )

    if "pipeline" in st.session_state:
        st.info("Documents: " + ", ".join(st.session_state.document_names))
        if st.button("Start a new document session"):
            for key in [
                "pipeline",
                "document_hash",
                "document_names",
                "messages",
            ]:
                st.session_state.pop(key, None)
            st.session_state.uploader_version += 1
            st.rerun()

if "pipeline" not in st.session_state:
    st.info("Upload a text-based PDF in the sidebar to start the demo.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.write(
                        f"{citation.get('source', 'Document')} — Page {citation['page']} "
                        f"(relevance: {citation['score']})"
                    )

question = st.chat_input("Ask something across your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                result = st.session_state.pipeline.ask(question)
            except Exception as exc:
                st.error(f"Could not answer this question: {exc}")
            else:
                st.write(result["answer"])
                if result.get("citations"):
                    with st.expander("Sources"):
                        for citation in result["citations"]:
                            st.write(
                                f"{citation.get('source', 'Document')} — Page {citation['page']} "
                                f"(relevance: {citation['score']})"
                            )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result.get("citations", []),
                    }
                )
