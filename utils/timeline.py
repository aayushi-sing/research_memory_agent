"""
timeline.py
Build and render the memory timeline from the document registry.
Tracks what was ingested, when, and what happened (upload / query / contradiction).
"""

from datetime import datetime
from typing import List, Dict
import json
from pathlib import Path

EVENTS_FILE = "./timeline_events.json"


# ── Event log ────────────────────────────────────────────────────────

def _load_events() -> List[Dict]:
    if Path(EVENTS_FILE).exists():
        with open(EVENTS_FILE) as f:
            return json.load(f)
    return []


def _save_events(events: List[Dict]):
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)


def log_event(event_type: str, label: str, detail: str = ""):
    """
    Log a timeline event.
    event_type: 'upload' | 'query' | 'contradiction' | 'connect'
    """
    events = _load_events()
    events.append({
        "type":      event_type,
        "label":     label,
        "detail":    detail,
        "timestamp": datetime.utcnow().isoformat(),
    })
    _save_events(events)


# ── Timeline builders ────────────────────────────────────────────────

def build_timeline_from_registry(registry: List[Dict]) -> List[Dict]:
    """
    Build timeline entries from the document registry.
    Each entry: {date_str, icon, label, sublabel}
    """
    entries = []
    for doc in registry:
        dt = datetime.fromisoformat(doc["uploaded_at"])
        entries.append({
            "timestamp": dt,
            "date_str":  dt.strftime("%b %d, %Y"),
            "time_str":  dt.strftime("%H:%M"),
            "icon":      "📄",
            "label":     f"Uploaded: {doc['filename']}",
            "sublabel":  f"{doc['chunks']} chunks stored in memory",
            "type":      "upload",
        })

    # Merge with logged events
    for ev in _load_events():
        dt = datetime.fromisoformat(ev["timestamp"])
        icon_map = {
            "query":        "🔍",
            "contradiction":"⚡",
            "connect":      "🔗",
            "upload":       "📄",
        }
        entries.append({
            "timestamp": dt,
            "date_str":  dt.strftime("%b %d, %Y"),
            "time_str":  dt.strftime("%H:%M"),
            "icon":      icon_map.get(ev["type"], "·"),
            "label":     ev["label"],
            "sublabel":  ev.get("detail", ""),
            "type":      ev["type"],
        })

    # Sort newest first
    entries.sort(key=lambda x: x["timestamp"], reverse=True)
    return entries


def render_timeline_html(entries: List[Dict]) -> str:
    """
    Render timeline entries as an HTML string for st.markdown / st.components.
    """
    if not entries:
        return "<p style='color:#8b949e; font-size:13px'>No events yet.</p>"

    COLOR_MAP = {
        "upload":        "#58a6ff",
        "query":         "#3fb950",
        "contradiction": "#f78166",
        "connect":       "#d2a8ff",
    }

    rows = []
    prev_date = None
    for e in entries:
        color = COLOR_MAP.get(e["type"], "#8b949e")

        # Date separator
        if e["date_str"] != prev_date:
            rows.append(
                f"<div style='font-size:11px; color:#8b949e; font-weight:600; "
                f"letter-spacing:0.08em; margin:12px 0 4px'>{e['date_str'].upper()}</div>"
            )
            prev_date = e["date_str"]

        rows.append(f"""
<div style='display:flex; gap:10px; align-items:flex-start; margin:4px 0; padding:8px 0;
            border-bottom:1px solid #21262d'>
  <div style='font-size:16px; flex-shrink:0'>{e['icon']}</div>
  <div>
    <div style='font-size:13px; color:#e6edf3; font-weight:500'>{e['label']}</div>
    <div style='font-size:11px; color:#8b949e; margin-top:2px'>{e['sublabel']}</div>
  </div>
  <div style='margin-left:auto; font-size:11px; color:#8b949e; flex-shrink:0'>{e['time_str']}</div>
</div>
""")

    return "".join(rows)