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

    external = obs.get("external_tracing", {})
    if external:
        st.header("🔗 External Tracing")
        cols = st.columns(2)

        lf = external.get("langfuse", {})
        with cols[0]:
            st.markdown("**Langfuse**")
            if lf.get("configured"):
                st.success("Configured")
                if lf.get("active"):
                    st.caption(f"Trace ID: `{lf.get('id')}`")
                if lf.get("url"):
                    st.markdown(f"[Open in Langfuse]({lf['url']})")
            else:
                st.info("Not configured — set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`")

        ls = external.get("langsmith", {})
        with cols[1]:
            st.markdown("**LangSmith**")
            if ls.get("configured"):
                st.success("Configured")
                if ls.get("active"):
                    st.caption(f"Run ID: `{ls.get('id')}`")
                if ls.get("url"):
                    st.markdown(f"[Open in LangSmith]({ls['url']})")
            else:
                st.info("Not configured — set `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true` in `.env`")

        total_ms = obs.get("total_duration_ms")
        if total_ms:
            st.metric("Total Endpoint Duration", f"{total_ms:.0f} ms")

        st.divider()

    # -----------------------------
    # Latency Metrics
    # -----------------------------
    durations = dict(obs.get("durations", {}))

    # OPTIONAL: Add DB spans to latency chart
    spans = obs.get("raw", {}).get("spans", [])
    db_spans = [s for s in spans if s.get("type") == "db"]
    for span in db_spans:
        durations[f"db:{span['name']}"] = span.get("duration_ms", 0)

    if durations:
        st.header("⏱️ Latency Metrics")
        df = pd.DataFrame([
            {"step": k, "latency_ms": v}
            for k, v in sorted(durations.items(), key=lambda item: item[1], reverse=True)
        ])
        fig = px.bar(df, x="step", y="latency_ms", title="Latency per Pipeline Step")
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
            inputs = span.get("inputs") or {}
            with st.expander(f"{span['name']} ({span.get('duration_ms', 0)} ms)"):
                st.json({
                    "query": span.get("query") or inputs.get("query"),
                    "collection": span.get("collection") or inputs.get("collection"),
                    "num_items": span.get("num_items") or inputs.get("num_items"),
                    "duration_ms": span.get("duration_ms"),
                    "error": span.get("error"),
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
