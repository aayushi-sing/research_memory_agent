"""
graph_builder.py
Build a knowledge graph from documents and shared concepts.
Uses networkx for graph logic and pyvis for HTML rendering.
"""

import networkx as nx
from pyvis.network import Network
from typing import List, Dict
from pathlib import Path
import hashlib

GRAPHS_DIR = "./graphs"

# Node color palette
DOC_COLOR     = "#58a6ff"   # Blue — document nodes
CONCEPT_COLOR = "#3fb950"   # Green — concept nodes
EDGE_COLOR    = "#30363d"   # Dark — edges


def build_graph(
    doc_concepts: Dict[str, List[str]],
    shared_concepts: Dict[str, List[str]],
) -> nx.Graph:
    """
    Build a networkx graph:
    - Document nodes (blue)
    - Shared concept nodes (green)
    - Edges: doc → concept
    """
    G = nx.Graph()

    # Add document nodes
    for doc in doc_concepts:
        short = doc.replace(".pdf", "").replace(".txt", "").replace(".md", "")
        G.add_node(doc, label=short, node_type="document", color=DOC_COLOR)

    # Add shared concept nodes + edges
    for concept, docs in shared_concepts.items():
        G.add_node(concept, label=concept, node_type="concept", color=CONCEPT_COLOR)
        for doc in docs:
            G.add_edge(doc, concept, weight=1)

    # Also add direct doc–doc edges when they share concepts
    docs_list = list(doc_concepts.keys())
    for i in range(len(docs_list)):
        for j in range(i + 1, len(docs_list)):
            a, b = docs_list[i], docs_list[j]
            shared = [
                c for c, d in shared_concepts.items()
                if a in d and b in d
            ]
            if shared:
                G.add_edge(a, b, weight=len(shared), label=f"{len(shared)} shared")

    return G


def render_graph_html(
    G: nx.Graph,
    filename: str = "graph.html",
    height: str = "400px",
) -> str:
    """
    Render a networkx graph to an interactive pyvis HTML file.
    Saves to ./graphs/{filename} and returns the full HTML string.
    """
    Path(GRAPHS_DIR).mkdir(exist_ok=True)
    output_path = f"{GRAPHS_DIR}/{filename}"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0d1117",
        font_color="#e6edf3",
        directed=False,
    )
    net.barnes_hut(spring_length=160, spring_strength=0.04)

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "concept")
        net.add_node(
            node,
            label=attrs.get("label", str(node)),
            color=attrs.get("color", CONCEPT_COLOR),
            size=28 if node_type == "document" else 14,
            shape="dot",
            title=f"{'📄 Document' if node_type == 'document' else '🔑 Concept'}: {node}",
        )

    for u, v, attrs in G.edges(data=True):
        net.add_edge(
            u, v,
            color=EDGE_COLOR,
            title=attrs.get("label", ""),
            width=1 + attrs.get("weight", 1) * 0.5,
        )

    net.set_options("""
    {
      "interaction": { "hover": true, "tooltipDelay": 100 },
      "physics": { "stabilization": { "iterations": 100 } }
    }
    """)

    net.save_graph(output_path)
    return open(output_path, encoding="utf-8").read()


def graph_stats(G: nx.Graph) -> Dict:
    """Return basic stats about the knowledge graph."""
    doc_nodes     = [n for n, d in G.nodes(data=True) if d.get("node_type") == "document"]
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "concept"]
    return {
        "total_nodes":    G.number_of_nodes(),
        "total_edges":    G.number_of_edges(),
        "documents":      len(doc_nodes),
        "shared_concepts": len(concept_nodes),
        "density":        round(nx.density(G), 4),
    }