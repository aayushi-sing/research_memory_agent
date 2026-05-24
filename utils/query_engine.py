"""
query_engine.py
Send retrieved chunks + user query to Groq LLaMA and return the answer.
Supports three reasoning modes: answer, connect, contradict.
"""

import os
from groq import Groq
from typing import List, Dict

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPTS = {
    "answer": """You are a Research Memory Agent with access to a semantic memory store.
Answer the user's question using ONLY the retrieved memory passages provided.
For each key insight, cite which document it came from using [Source: filename].
End your response with a bold '## Memory Synthesis' section that summarises the core takeaway in 2-3 sentences.""",

    "connect": """You are a research synthesis expert analysing a set of academic documents.
Your task: identify recurring themes, shared concepts, and conceptual bridges across the retrieved passages.
Structure your response as:
1. **Recurring Themes** — ideas that appear in multiple sources
2. **Conceptual Bridges** — how the ideas connect across papers
3. **Emerging Pattern** — the meta-insight that spans all sources
Always cite which documents share each idea.""",

    "contradict": """You are a critical research analyst.
Your task: detect contradictions, disagreements, or tensions between the retrieved passages.
Structure your response as:
- Start with 'CONTRADICTION DETECTED:' or 'NO CONTRADICTION FOUND' on the first line.
- List each conflict with: Claim A (source) vs Claim B (source) — Explanation
- End with a brief note on why these contradictions might exist (different methodology, time period, scope, etc.)""",
}


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    parts = []
    for c in chunks:
        parts.append(
            f"[Source: {c['filename']} | chunk {c['chunk_index']} | relevance {int(c['score']*100)}%]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def run_query(query: str, chunks: List[Dict], mode: str = "answer") -> str:
    """
    Send query + retrieved context to Groq and return the answer string.
    mode: 'answer' | 'connect' | 'contradict'
    """
    if not chunks:
        return "No relevant memory found. Try uploading more documents or rephrasing your query."

    if mode not in SYSTEM_PROMPTS:
        mode = "answer"

    context = build_context(chunks)
    client  = _get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {
                "role": "user",
                "content": f"User Query: {query}\n\n=== Retrieved Memory ===\n\n{context}",
            },
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content