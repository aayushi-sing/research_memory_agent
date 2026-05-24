"""
reasoning.py
Use spaCy NLP to extract key concepts, named entities, and noun phrases
from text chunks. Powers the graph builder and concept overlay.
"""

from typing import List, Dict
import spacy

# We use the small English model — fast and free
# If not installed: python -m spacy download en_core_web_sm
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def extract_concepts(text: str, top_n: int = 10) -> List[str]:
    """
    Extract the most meaningful noun phrases and named entities from text.
    Returns a deduplicated list of concept strings (lowercased).
    """
    nlp  = _get_nlp()
    doc  = nlp(text[:5000])  # Limit for speed

    concepts = set()

    # Named entities (people, orgs, technologies, etc.)
    for ent in doc.ents:
        if ent.label_ not in ("CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"):
            concepts.add(ent.text.lower().strip())

    # Noun chunks (key phrases like "transformer architecture")
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        # Filter out short/generic phrases
        if len(phrase) > 3 and not phrase.startswith(("the ", "a ", "an ", "this ", "that ", "these ")):
            concepts.add(phrase)

    # Sort by length desc (longer = more specific), return top_n
    sorted_concepts = sorted(concepts, key=len, reverse=True)
    return sorted_concepts[:top_n]


def extract_concepts_from_chunks(chunks: List[Dict]) -> Dict[str, List[str]]:
    """
    Extract concepts for each unique document in the chunk list.
    Returns: {filename: [concept, concept, ...]}
    """
    doc_texts: Dict[str, str] = {}
    for c in chunks:
        fn = c["filename"]
        doc_texts[fn] = doc_texts.get(fn, "") + " " + c["text"]

    return {fn: extract_concepts(text) for fn, text in doc_texts.items()}


def find_shared_concepts(doc_concepts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Find concepts that appear in more than one document.
    Returns: {concept: [doc1, doc2, ...]}
    """
    concept_docs: Dict[str, List[str]] = {}
    for doc, concepts in doc_concepts.items():
        for concept in concepts:
            concept_docs.setdefault(concept, []).append(doc)

    # Keep only concepts shared by 2+ documents
    return {c: docs for c, docs in concept_docs.items() if len(docs) > 1}