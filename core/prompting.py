"""Prompt construction for grounded, multi-turn document questions."""


def build_rag_prompt(context: str, question: str, history: list[dict]) -> str:
    """Build a grounded prompt using at most the three latest conversation turns."""
    recent = history[-3:]
    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in recent
    )
    return f"""You are a helpful assistant that answers questions about a document.
Only use the context provided below. If the answer isn't there, say so.
Always mention which page the information is from.

Previous conversation:
{history_text if history_text else 'None'}

Document context:
{context}

Question: {question}

Answer:"""
