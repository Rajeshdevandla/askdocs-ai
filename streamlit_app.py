import streamlit as st
import time
import random

# ── Demo mode: no AWS credentials needed ──────────────────────────────────────
DEMO_RESPONSES = {
      "revenue":    ("Total revenue in Q3 2024 was **$4.2 billion**, a 12% increase year-over-year.\n\n> *Source: Page 14, Financial Highlights section*", [14, 15]),
      "risk":       ("The report identifies three primary risk factors:\n1. Macroeconomic uncertainty\n2. Supply chain disruptions\n3. Regulatory changes in the EU market\n\n> *Source: Page 31, Risk Factors section*", [31, 32]),
      "summary":    ("This document provides a comprehensive overview of the company's financial performance, strategic initiatives, and market outlook for fiscal year 2024.\n\n> *Source: Page 2, Executive Summary*", [2, 3]),
      "employees":  ("As of Q3 2024, the company employs **12,400 full-time employees** across 18 countries.\n\n> *Source: Page 22, Human Capital section*", [22]),
      "default":    ("Based on the document, I found relevant information in several sections. The content covers financial performance, operational metrics, and strategic direction.\n\n> *Source: Pages 5-8, Core Analysis*", [5, 6, 7]),
}

def get_demo_answer(question: str) -> tuple[str, list[int]]:
      q = question.lower()
      for keyword, (answer, pages) in DEMO_RESPONSES.items():
                if keyword in q:
                              return answer, pages
                      return DEMO_RESPONSES["default"]

  def stream_text(text: str):
        for word in text.split(" "):
                  yield word + " "
                  time.sleep(0.03)

    # ── Page config ───────────────────────────────────────────────────────────────
    st.set_page_config(page_title="AskDocs AI", page_icon="📄", layout="centered")

st.title("📄 AskDocs AI")
st.caption("Upload a PDF and ask questions — powered by Amazon Bedrock (Claude 3 Haiku) + FAISS")
st.info("🎭 **Demo mode** — responses are illustrative. [Deploy with your AWS keys](https://github.com/Rajeshdevandla/askdocs-ai) for full RAG.", icon="ℹ️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
      st.header("📁 Upload a Document")
      uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and "session_id" not in st.session_state:
              with st.spinner("Extracting text and building FAISS index..."):
                            time.sleep(2.0)
                        st.session_state.session_id = "demo-" + str(random.randint(1000, 9999))
        st.session_state.document_name = uploaded_file.name
        st.session_state.page_count = random.randint(18, 45)
        st.session_state.chunk_count = st.session_state.page_count * 6
        st.session_state.messages = []
        st.success(f"✅ Ready! {st.session_state.page_count} pages, {st.session_state.chunk_count} chunks indexed.")

    if "session_id" in st.session_state:
              st.markdown("---")
        st.markdown(f"**Document:** {st.session_state.document_name}")
        st.markdown(f"**Pages:** {st.session_state.page_count}")
        st.markdown(f"**Chunks:** {st.session_state.chunk_count}")
        if st.button("🔄 Upload new document"):
                      for key in ["session_id", "document_name", "messages", "page_count", "chunk_count"]:
                                        st.session_state.pop(key, None)
                                    st.rerun()

    st.markdown("---")
    st.markdown("**Try asking:**")
    st.markdown("- What was the total revenue?")
    st.markdown("- What are the key risk factors?")
    st.markdown("- Summarize this document")
    st.markdown("- How many employees?")

# ── Main chat area ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
      st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
              st.markdown("**1️⃣ Upload**\nDrop any PDF in the sidebar")
    with col2:
              st.markdown("**2️⃣ Index**\nFAISS + sentence-transformers build the vector index locally")
    with col3:
              st.markdown("**3️⃣ Ask**\nClaude 3 Haiku answers with page-level citations")
    st.stop()

if "messages" not in st.session_state:
      st.session_state.messages = []

for msg in st.session_state.messages:
      with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if msg.get("pages"):
                      with st.expander("📎 Sources"):
                                        for p in msg["pages"]:
                                                              st.markdown(f"- Page {p}")

                        question = st.chat_input("Ask something about your document...")

if question:
      st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
              st.markdown(question)

    with st.chat_message("assistant"):
              with st.spinner("Searching FAISS index → calling Bedrock..."):
                            time.sleep(1.2)
                        answer, pages = get_demo_answer(question)
        st.write_stream(stream_text(answer))
        with st.expander("📎 Sources"):
                      for p in pages:
                                        st.markdown(f"- Page {p}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "pages": pages})
