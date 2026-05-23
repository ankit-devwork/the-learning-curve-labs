import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
import nest_asyncio

from latency_radar import app

nest_asyncio.apply()

st.set_page_config(page_title="Latency Radar v1.0", layout="wide")
st.title("⚡ AI Latency Radar Dashboard (v1.0)")

prompt = st.text_area("Benchmark Prompt", "Explain atomic resonance in one sentence.")

models = st.multiselect(
    "Select Models",
    ["gpt-3.5-turbo", "groq/llama-3.3-70b-versatile"],
    default=["gpt-3.5-turbo", "groq/llama-3.3-70b-versatile"]
)

async def run_radar(state):
    return await app.ainvoke(state)

if st.button("Run Benchmark"):
    with st.spinner("Running Latency Radar v5.5..."):
        state = {
            "models": models,
            "prompt": prompt,
            "results": {},
            "judge": {"winner_model": None, "ranking": [], "reasoning": "", "confidence": 0},
            "summary": {},
            "metadata": {},
            "scores": {},
        }

        loop = asyncio.get_event_loop()
        final = loop.run_until_complete(run_radar(state))

    results = final["results"]
    judge = final["judge"]
    metadata = final["metadata"]
    summary = final["summary"]
    scores = final["scores"]

    # Error display
    failed = [r for r in results.values() if r["status"] == "error"]
    if failed:
        st.error("Some models failed:")
        for f in failed:
            st.write(f)

    # Build DataFrame
    rows = []
    for model, r in results.items():
        if r["status"] == "success":
            meta = metadata.get(model, {})
            score = scores.get(model, 0)

            rows.append({
                "model": model,
                "cold_ttft_ms": r.get("cold_ttft_ms"),
                "hot_ttft_ms": r["ttft_ms"],
                "total_ms": r["total_ms"],
                "tps": r["tps"],
                "cost_usd": r["cost_usd"],
                "radar_score": score,
                "provider": meta.get("provider", "Unknown"),
                "context_window": meta.get("context_window", "Unknown"),
                "max_output_tokens": meta.get("max_output_tokens", "Unknown"),
                "description": meta.get("description", "No description available.")
            })

    df = pd.DataFrame(rows)

    st.subheader("📊 Radar Scores & Latency")
    st.dataframe(df)

    fig = px.bar(df, x="model", y="radar_score", color="provider", title="Radar Score Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Hot TTFT vs TPS")
    fig2 = px.scatter(df, x="hot_ttft_ms", y="tps", color="model", size="radar_score")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧠 Judge Evaluation")
    st.write(judge)

    st.subheader("📦 Model Metadata")
    st.json(metadata)

    st.subheader("🏁 Summary")
    st.json(summary)
