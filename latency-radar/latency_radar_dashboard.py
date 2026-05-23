import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio

from latency_radar import app  # <- updated backend

st.set_page_config(page_title="Latency Radar v5.3", layout="wide")

st.title("⚡ AI Latency Radar Dashboard (v5.3)")

prompt = st.text_area("Benchmark Prompt", "Explain atomic resonance in one sentence.")

models = st.multiselect(
    "Select Models",
    ["gpt-3.5-turbo", "groq/llama-3.3-70b-versatile"],
    default=["gpt-3.5-turbo", "groq/llama-3.3-70b-versatile"]
)

if st.button("Run Benchmark"):
    with st.spinner("Running Latency Radar v5.3..."):
        state = {
            "models": models,
            "prompt": prompt,
            "results": {},
            "judge": {"winner_model": None, "ranking": [], "reasoning": "", "confidence": 0},
            "summary": {},
            "metadata": {},
            "scores": {},
        }

        final = asyncio.run(app.ainvoke(state))

    results = final["results"]
    judge = final["judge"]
    metadata = final["metadata"]
    summary = final["summary"]
    scores = final["scores"]

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
