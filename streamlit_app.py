import streamlit as st
import time
import random

DEMO_RESPONSES = {
    "revenue":   ("Total revenue in Q3 2024 was $4.2 billion, a 12% increase year-over-year.\n\nSource: Page 14.", [14, 15]),
    "risk":      ("Three primary risk factors:\n1. Macroeconomic uncertainty\n2. Supply chain disruptions\n3. Regulatory changes.\n\nSource: Page 31.", [31, 32]),
    "summary":   ("This document covers financial performance, strategic initiatives, and market outlook for 2024.\n\nSource: Page 1.", [1, 2]),
    "employees": ("As of Q3 2024, the company employs 12,400 employees across 18 countries.\n\nSource: Page 22.", [22]),
    "default":   ("I found relevant information covering financial performance and strategic outlook.\n\nSource: Pages 1-5.", [1, 2, 3]),
}

def get_demo_answer(question: str):
    q = question.lower()
    for keyword, (answer, pages) in DEMO_RESPONSES.items():
        if keyword in q:
            return answer, pages
    return DEMO_RESPONSES["default"]

def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

st.set_page_config(page_title="AskDocs AI", page_icon="\U0001f4c4", layout="centered")
st.title("\U0001f4c4 AskDocs AI")
st.caption("Upload a PDF and ask questions - powered by Amazon Bedrock + FAISS")
st.info("Demo mode - responses are illustrative.", icon="\U0001f4a1")

with st.sidebar:
    st.header("\U0001f4c2 Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing..."):
            time.sleep(2)
        st.success(f"\u2705 {uploaded_file.name} processed!")
    else:
        st.caption("Upload a PDF to get started.")
    st.divider()
    st.markdown("**Stack**")
    st.markdown("- FAISS vector search")
    st.markdown("- Amazon Bedrock Claude 3")
    st.markdown("- Page-level citations")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me about **revenue**, **risk**, **employees**, or **summary**."
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer, pages = get_demo_answer(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            time.sleep(1)
        st.write_stream(stream_text(answer))
        with st.expander(f"\U0001f4cd Sources - Pages {pages}"):
            for p in pages:
                st.markdown(f"- Page **{p}**: Relevant excerpt found.")
    st.session_state.messages.append({"role": "assistant", "content": answer})
