'''loading'''

import streamlit as st
import os
import uuid
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

load_dotenv()  # loads GROQ_API_KEY from .env

# ── Util imports ─────────────────────────────────────────────────────
from utils.pdf_loader    import extract_text
from utils.chunker       import chunk_with_metadata
from utils.memory_store  import (
    add_document, register_document, delete_document,
    query_chunks, load_registry, total_chunks,
)
from utils.query_engine  import run_query
from utils.reasoning     import extract_concepts_from_chunks, find_shared_concepts
from utils.graph_builder import build_graph, render_graph_html, graph_stats
from utils.timeline      import build_timeline_from_registry, render_timeline_html, log_event

# ── Ensure folders exist ─────────────────────────────────────────────
Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)
Path("graphs").mkdir(exist_ok=True)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Memory Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* === Global dark theme === */
  [data-testid="stAppViewContainer"]   { background:#0d1117; color:#e6edf3; }
  [data-testid="stSidebar"]            { background:#161b22; border-right:1px solid #21262d; }
  [data-testid="stSidebar"] *          { color:#e6edf3 !important; }
  [data-testid="stHeader"]             { background:#0d1117; }
  section[data-testid="stSidebar"] > div { padding-top: 1rem; }

  /* === Inputs === */
  .stTextInput > div > div > input,
  .stTextArea  > div > div > textarea {
    background:#161b22 !important; color:#e6edf3 !important;
    border:1px solid #21262d !important; border-radius:8px !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea  > div > div > textarea:focus {
    border-color:#58a6ff !important; box-shadow:none !important;
  }

  /* === Buttons === */
  .stButton > button {
    background:#238636 !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
  }
  .stButton > button:hover  { background:#2ea043 !important; }
  .stButton > button:active { transform:scale(0.98); }

  /* === File uploader === */
  [data-testid="stFileUploader"] {
    background:#161b22 !important; border:1px dashed #30363d !important; border-radius:8px !important;
  }

  /* === Radio / selectbox === */
  div[data-testid="stRadio"] label span  { color:#e6edf3 !important; }
  div[data-baseweb="select"] * { background:#161b22 !important; color:#e6edf3 !important; }

  /* === Tabs === */
  [data-testid="stTabs"] button          { color:#8b949e !important; border-radius:6px 6px 0 0; }
  [data-testid="stTabs"] button[aria-selected="true"] { color:#e6edf3 !important; border-bottom:2px solid #58a6ff !important; }

  /* === Custom components === */
  .mem-card {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:18px 22px; margin:8px 0;
  }
  .chunk-card {
    background:#0d1117; border:1px solid #21262d;
    border-left:3px solid #238636; border-radius:6px;
    padding:12px 14px; margin:6px 0; font-size:13px; color:#8b949e;
  }
  .badge-score  { background:#0d2818; color:#3fb950; border:1px solid #238636; border-radius:20px; padding:2px 10px; font-size:12px; font-weight:700; display:inline-block; margin-right:6px; }
  .badge-source { background:#0c1e3a; color:#58a6ff;  border:1px solid #1f4e8c; border-radius:20px; padding:2px 10px; font-size:12px; display:inline-block; margin-right:4px; }
  .badge-mode   { background:#2d1b4e; color:#d2a8ff;  border:1px solid #6e40c9; border-radius:20px; padding:2px 10px; font-size:12px; display:inline-block; }

  .log-box {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:14px 18px; font-family:monospace; font-size:13px;
  }
  .stat-card {
    background:#161b22; border:1px solid #21262d; border-radius:8px;
    padding:14px 16px; text-align:center;
  }
  h1,h2,h3 { color:#e6edf3 !important; }
  p,li      { color:#c9d1d9; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Research Memory")
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("➕ Ingest to Memory", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one file first.")
        else:
            registry_names = [d["filename"] for d in load_registry()]
            for f in uploaded_files:
                if f.name in registry_names:
                    st.info(f"Already in memory: {f.name}")
                    continue
                with st.spinner(f"Processing {f.name}…"):
                    # Save to disk
                    save_path = f"uploads/{f.name}"
                    with open(save_path, "wb") as fh:
                        fh.write(f.read())
                    try:
                        text   = extract_text(save_path)
                        doc_id = str(uuid.uuid4())
                        chunks = chunk_with_metadata(text, f.name, doc_id)
                        add_document(doc_id, chunks)
                        register_document(doc_id, f.name, len(chunks), text)
                        log_event("upload", f"Uploaded: {f.name}", f"{len(chunks)} chunks")
                        st.success(f"✓ {f.name}  ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.rerun()

    st.markdown("---")

    # ── Memory Bank ───────────────────────────────────────────────────
    st.markdown("### Memory Bank")
    registry = load_registry()

    if not registry:
        st.markdown("<small style='color:#8b949e'>No documents yet.</small>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<span style='color:#3fb950; font-size:13px'>**{len(registry)} doc(s)** · "
            f"**{total_chunks()} chunks** in memory</span>",
            unsafe_allow_html=True,
        )
        for doc in registry:
            from datetime import datetime
            dt = datetime.fromisoformat(doc["uploaded_at"]).strftime("%b %d")
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(
                    f"<div style='font-size:12px; padding:3px 0'>"
                    f"<span style='color:#58a6ff'>{dt}</span> "
                    f"<span style='color:#e6edf3'>📄 {doc['filename']}</span><br>"
                    f"<span style='color:#8b949e'>{doc['chunks']} chunks</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"del_{doc['doc_id']}"):
                    delete_document(doc["doc_id"])
                    fp = Path(f"uploads/{doc['filename']}")
                    if fp.exists():
                        fp.unlink()
                    st.rerun()

    st.markdown("---")

    # ── Timeline ──────────────────────────────────────────────────────
    st.markdown("### Memory Timeline")
    timeline = build_timeline_from_registry(load_registry())
    if timeline:
        for e in timeline[:8]:
            color_map = {
                "upload": "#58a6ff", "query": "#3fb950",
                "contradiction": "#f78166", "connect": "#d2a8ff",
            }
            color = color_map.get(e["type"], "#8b949e")
            st.markdown(
                f"<div style='font-size:11px; color:#8b949e; margin:3px 0'>"
                f"<span style='color:{color}'>{e['icon']}</span> "
                f"<span style='color:{color}'>{e['date_str']}</span> — "
                f"{e['label']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<small style='color:#8b949e'>Nothing logged yet.</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════
st.markdown("# Research Memory Agent")
st.markdown(
    "<p style='color:#8b949e; margin-top:-10px'>"
    "Persistent semantic memory · contradiction detection · idea synthesis · knowledge graphs"
    "</p>",
    unsafe_allow_html=True,
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

registry = load_registry()
if not registry:
    st.warning("👈 Upload at least one PDF or text file using the sidebar to start.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────
tab_query, tab_graph, tab_timeline = st.tabs(["🔍 Query Memory", "🕸 Knowledge Graph", "🕐 Timeline"])


# ════════════════════════════════
# TAB 1 — QUERY
# ════════════════════════════════
with tab_query:

    # Mode selector
    st.markdown("#### Query Mode")
    mode_col1, mode_col2, mode_col3 = st.columns(3)

    MODES = {
        "answer":     ("💡", "Answer",          "Semantic search + synthesis"),
        "connect":    ("🔗", "Connect Ideas",   "Find recurring themes"),
        "contradict": ("⚡", "Contradictions",  "Detect disagreements"),
    }

    if "mode" not in st.session_state:
        st.session_state.mode = "answer"

    for col, (mid, (icon, label, desc)) in zip(
        [mode_col1, mode_col2, mode_col3], MODES.items()
    ):
        with col:
            active = st.session_state.mode == mid
            if st.button(
                f"{icon} {label}",
                key=f"mode_{mid}",
                use_container_width=True,
                help=desc,
            ):
                st.session_state.mode = mid

    mode = st.session_state.mode
    icon, label, desc = MODES[mode]
    st.markdown(
        f"<small style='color:#8b949e'>Mode: <span class='badge-mode'>{icon} {label}</span> — {desc}</small>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Ask Your Memory")
    placeholders = {
        "answer":     "What is the main argument across these papers?",
        "connect":    "What ideas repeatedly appear across these documents?",
        "contradict": "Which papers disagree on this topic?",
    }

    # Suggestion chips rendered BEFORE the text_input widget
    suggestions = [
        "What ideas repeat across all papers?",
        "Which sources contradict each other?",
        "What is the key contribution of each document?",
        "Summarise the most important findings",
    ]
    st.markdown("<small style='color:#8b949e'>Suggestions:</small>", unsafe_allow_html=True)
    scols = st.columns(4)
    for i, s in enumerate(suggestions):
        with scols[i]:
            if st.button(s, key=f"s{i}"):
                st.session_state["pending_query"] = s
                st.rerun()

    # Read pending suggestion (set by chip click above) BEFORE widget renders
    default_val = st.session_state.pop("pending_query", "")

    query = st.text_input(
    "Search query", placeholder=placeholders[mode],
    value=default_val,
    label_visibility="collapsed",
    )

    run_btn = st.button("Search Memory", use_container_width=True)

    if run_btn and query.strip():
        log_placeholder = st.empty()
        STEPS = [
            ("🔵", "Encoding query with sentence-transformers"),
            ("🔵", "Searching semantic memory in ChromaDB"),
            ("🔵", "Extracting NLP concepts with spaCy"),
            ("🔵", "Retrieving top-k relevant chunks"),
            ("🔵", "Sending to LLaMA 3.3 70B via Groq"),
            ("🔵", "Synthesising answer"),
        ]

        def render_log(done_up_to: int):
            lines = []
            for i, (_, step) in enumerate(STEPS):
                if i < done_up_to:
                    lines.append(f"<div style='color:#3fb950'>✓ {step}</div>")
                elif i == done_up_to:
                    lines.append(f"<div style='color:#e6edf3'>· {step}…</div>")
                else:
                    lines.append(f"<div style='color:#30363d'>· {step}</div>")
            log_placeholder.markdown(
                "<div class='log-box'>" + "".join(lines) + "</div>",
                unsafe_allow_html=True,
            )

        # ── Run with animated progress ────────────────────────────────
        for step_i in range(len(STEPS)):
            render_log(step_i)
            time.sleep(0.22)

        chunks = query_chunks(query, top_k=10)
        log_placeholder.empty()

        if not chunks:
            st.error("No relevant memory found. Try uploading more documents.")
            st.stop()

        answer = run_query(query, chunks, mode)
        log_event(mode, f"{icon} {query[:60]}", f"{len(chunks)} chunks retrieved")

        st.markdown("---")

        # ── Answer ────────────────────────────────────────────────────
        st.markdown(f"### {icon} Memory Synthesis")
        st.markdown(f"<div class='mem-card'>{answer}</div>", unsafe_allow_html=True)

        # Source badges
        source_chips = " ".join(
            f"<span class='badge-score'>{int(c['score']*100)}%</span>"
            f"<span class='badge-source'>{c['filename'].replace('.pdf','').replace('.txt','')}</span>"
            for c in chunks
        )
        st.markdown(f"<div style='margin:10px 0'>{source_chips}</div>", unsafe_allow_html=True)

        # ── Concept extraction panel ──────────────────────────────────
        st.markdown("### Key Concepts Detected")
        doc_concepts   = extract_concepts_from_chunks(chunks)
        shared_concepts = find_shared_concepts(doc_concepts)

        if shared_concepts:
            concept_cols = st.columns(min(4, len(shared_concepts)))
            for i, (concept, docs) in enumerate(list(shared_concepts.items())[:8]):
                with concept_cols[i % len(concept_cols)]:
                    st.markdown(
                        f"<div style='background:#161b22; border:1px solid #21262d; border-radius:8px; "
                        f"padding:10px 12px; margin:4px 0; font-size:12px'>"
                        f"<div style='color:#d2a8ff; font-weight:600'>{concept}</div>"
                        f"<div style='color:#8b949e; margin-top:4px'>"
                        f"Found in {len(docs)} doc(s)</div></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("<small style='color:#8b949e'>No cross-document concepts found in this result set.</small>", unsafe_allow_html=True)

        # ── Inline mini-graph ─────────────────────────────────────────
        unique_sources = list({c["filename"] for c in chunks})
        if len(unique_sources) > 1:
            st.markdown("### 🕸 Relationship Graph (this query)")
            G = build_graph(doc_concepts, shared_concepts)
            graph_html = render_graph_html(G, filename=f"query_{abs(hash(query))}.html", height="320px")
            if graph_html:
                st.iframe(graph_html, height=340)

        # ── Retrieved chunks ──────────────────────────────────────────
        with st.expander(f" View {len(chunks)} retrieved memory chunks"):
            for c in chunks:
                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<span class='badge-score'>{int(c['score']*100)}%</span>"
                    f"<span class='badge-source'>{c['filename']}</span>"
                    f" chunk {c['chunk_index']}<br><br>{c['text'][:400]}…"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    elif run_btn:
        st.warning("Please enter a query first.")


# ════════════════════════════════
# TAB 2 — KNOWLEDGE GRAPH
# ════════════════════════════════
with tab_graph:
    st.markdown("### 🕸 Full Knowledge Graph")
    st.markdown(
        "<p style='color:#8b949e'>All documents and their shared concepts visualised as an interactive graph.</p>",
        unsafe_allow_html=True,
    )

    registry = load_registry()
    if len(registry) < 2:
        st.info("Upload at least 2 documents to see the knowledge graph.")
    else:
        if st.button("🔄 Regenerate Graph", use_container_width=False):
            st.session_state.pop("full_graph_html", None)

        if "full_graph_html" not in st.session_state:
            with st.spinner("Extracting concepts and building graph…"):
                # Pull all stored chunks for concept extraction
                all_chunks_raw = []
                for doc in registry:
                    sample = query_chunks(
                        doc["filename"].replace(".pdf","").replace(".txt",""),
                        top_k=20,
                    )
                    all_chunks_raw.extend(sample)

                doc_concepts    = extract_concepts_from_chunks(all_chunks_raw)
                shared_concepts = find_shared_concepts(doc_concepts)
                G               = build_graph(doc_concepts, shared_concepts)
                stats           = graph_stats(G)
                html            = render_graph_html(G, "full_graph.html", "500px")
                st.session_state.full_graph_html  = html
                st.session_state.full_graph_stats = stats

        # Stats bar
        stats = st.session_state.get("full_graph_stats", {})
        if stats:
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, (label, val) in zip(
                [sc1, sc2, sc3, sc4],
                [
                    ("Documents",       stats.get("documents", 0)),
                    ("Shared Concepts", stats.get("shared_concepts", 0)),
                    ("Graph Edges",     stats.get("total_edges", 0)),
                    ("Density",         stats.get("density", 0)),
                ],
            ):
                with col:
                    st.markdown(
                        f"<div class='stat-card'>"
                        f"<div style='font-size:22px; font-weight:700; color:#58a6ff'>{val}</div>"
                        f"<div style='font-size:12px; color:#8b949e; margin-top:4px'>{label}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("")

        graph_html = st.session_state.get("full_graph_html", "")
        if graph_html:
            st.iframe(graph_html, height=520)

        st.markdown(
            "<small style='color:#8b949e'>💡 Drag nodes · Hover for details · Scroll to zoom</small>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════
# TAB 3 — TIMELINE
# ════════════════════════════════
with tab_timeline:
    st.markdown("###  Memory Timeline")
    st.markdown(
        "<p style='color:#8b949e'>Every upload, query, and detected event — in chronological order.</p>",
        unsafe_allow_html=True,
    )

    registry = load_registry()
    timeline  = build_timeline_from_registry(registry)

    if not timeline:
        st.info("No timeline events yet. Upload documents and run queries to build history.")
    else:
        # Filter controls
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            type_filter = st.multiselect(
                "Filter by type",
                options=["upload", "query", "connect", "contradiction"],
                default=["upload", "query", "connect", "contradiction"],
                label_visibility="collapsed",
            )
        with filter_col2:
            st.markdown(f"<small style='color:#8b949e'>{len(timeline)} events</small>", unsafe_allow_html=True)

        filtered = [e for e in timeline if e["type"] in type_filter]
        html     = render_timeline_html(filtered)
        st.markdown(
            f"<div style='background:#161b22; border:1px solid #21262d; border-radius:10px; "
            f"padding:16px 20px; max-height:600px; overflow-y:auto'>{html}</div>",
            unsafe_allow_html=True,
        )