import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="RAG Observability Dashboard", page_icon="📊")

st.title("📊 RAG Observability Dashboard")
st.caption("End‑to‑end visibility for your LangGraph multi‑agent pipeline")

# ---------------------------------------------------------
# Initialize Observability History
# ---------------------------------------------------------
if "observability_history" not in st.session_state:
    st.session_state.observability_history = []


# ---------------------------------------------------------
# Helper: Render a single observability record
# ---------------------------------------------------------

def _normalize_observability(obs):
    if isinstance(obs, str):
        try:
            return json.loads(obs)
        except json.JSONDecodeError:
            return {"raw": obs}

    if obs is None:
        return {}

    if not isinstance(obs, dict):
        return {"raw": str(obs)}

    return obs


def render_observability_record(record):
    obs = _normalize_observability(record["observability"])

    st.subheader(f"🔍 Trace: {record['label']}")
    st.json({
        "endpoint": record["endpoint"],
        "correlation_id": record["correlation_id"],
        "model": record.get("model"),
        "answer": record.get("answer"),
    })

    st.divider()

    # -----------------------------
    # Latency Metrics
    # -----------------------------
    durations = obs.get("durations", {})

    # OPTIONAL: Add DB spans to latency chart
    spans = obs.get("raw", {}).get("spans", [])
    db_spans = [s for s in spans if s.get("type") == "db"]
    for span in db_spans:
        durations[f"db:{span['name']}"] = span.get("duration_ms", 0)

    if durations:
        st.header("⏱️ Latency Metrics")
        df = pd.DataFrame([
            {"step": k, "latency_ms": v}
            for k, v in durations.items()
        ])
        fig = px.bar(df, x="step", y="latency_ms", title="Latency per Agent Step")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Retrieval Inspector
    # -----------------------------
    retr = obs.get("retriever")
    if retr:
        st.header("📚 Retrieval Inspector")
        st.json(retr)

    # -----------------------------
    # Document Selection Inspector
    # -----------------------------
    sel = obs.get("selection")
    if sel:
        st.header("📄 Document Selection Inspector")
        st.json(sel)

    # -----------------------------
    # Hallucination Metrics
    # -----------------------------
    hall = obs.get("hallucination")
    if hall and hall.get("score") is not None:
        st.header("🧠 Hallucination Metrics")
        score = hall["score"]
        st.metric("Similarity Score", f"{score:.2f}")
        st.progress(score)

    # -----------------------------
    # DB Operations Inspector
    # -----------------------------
    if db_spans:
        st.header("🗄️ Database Operations")
        for span in db_spans:
            with st.expander(f"{span['name']} ({span.get('duration_ms', 0)} ms)"):
                st.json({
                    "query": span.get("query"),
                    "collection": span.get("collection"),
                    "num_items": span.get("num_items"),
                    "duration_ms": span.get("duration_ms"),
                    "error": span.get("error")
                })

    # -----------------------------
    # Raw Trace
    # -----------------------------
    st.header("📊 Raw Trace")
    st.json(obs.get("raw", obs))


# ---------------------------------------------------------
# Sidebar: Observability History
# ---------------------------------------------------------
st.sidebar.title("📜 Observability History")

if st.sidebar.button("Clear History"):
    st.session_state.observability_history = []
    st.rerun()

if st.session_state.observability_history:
    labels = [rec["label"] for rec in st.session_state.observability_history]
    selected = st.sidebar.selectbox("Select a trace to inspect:", labels)
    record = next(r for r in st.session_state.observability_history if r["label"] == selected)
    render_observability_record(record)
else:
    st.sidebar.info("No observability traces yet.")
