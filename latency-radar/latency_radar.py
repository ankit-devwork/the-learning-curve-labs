import asyncio
import time
from typing import Dict, List, TypedDict, Literal, Optional, Any

from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
import litellm
from litellm import token_counter
from langfuse.langchain import CallbackHandler
from langsmith import traceable
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# Optional: provider-specific tokenizers (HF)
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # graceful fallback

load_dotenv()
langfuse_handler = CallbackHandler()
litellm.set_verbose = False

# =========================
# TYPES
# =========================

class ProbeResult(TypedDict, total=False):
    status: Literal["success", "error"]
    model: str
    content: str
    # hot metrics
    ttft_ms: float
    total_ms: float
    completion_ms: float
    tps: float
    cost_usd: float
    # cold metrics
    cold_ttft_ms: float
    cold_total_ms: float
    error: Optional[str]


class JudgeResult(TypedDict, total=False):
    winner_model: Optional[str]
    ranking: List[str]
    reasoning: str
    confidence: int


class RadarState(TypedDict):
    models: List[str]
    prompt: str
    results: Dict[str, ProbeResult]
    judge: JudgeResult
    summary: Dict[str, Any]
    metadata: Dict[str, Dict[str, Any]]
    scores: Dict[str, float]


# =========================
# METADATA + PRICING (HYBRID)
# =========================

MODEL_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 60 * 60 * 24  # 24 hours

PROVIDER_FALLBACK = {
    "openai": {
        "provider": "OpenAI",
        "description": "OpenAI GPT-series model.",
    },
    "groq": {
        "provider": "Groq",
        "description": "Groq LPU-accelerated model optimized for ultra-low latency.",
    },
    "anthropic": {
        "provider": "Anthropic",
        "description": "Claude model family focused on reasoning and safety.",
    },
    "mistral": {
        "provider": "Mistral AI",
        "description": "Mistral models optimized for speed and efficiency.",
    },
    "deepseek": {
        "provider": "DeepSeek",
        "description": "DeepSeek models optimized for cost-efficient reasoning.",
    },
}

# Fallback pricing table (per-token, dollars)
PRICING_TABLE = {
    "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},  # $0.50 / $1.50 per 1K
    "gpt-4o": {"prompt": 0.005 / 1000, "completion": 0.015 / 1000},           # $5 / $15 per 1M
    "groq/llama-3.3-70b-versatile": {"prompt": 0.59 / 1_000_000, "completion": 0.79 / 1_000_000},
    "groq/llama3-8b": {"prompt": 0.05 / 1_000_000, "completion": 0.05 / 1_000_000},
}


def detect_provider(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/")[0].lower()
    if model_name.startswith("gpt"):
        return "openai"
    return "unknown"


def get_hybrid_pricing(model_name: str) -> Dict[str, float]:
    """
    Try LiteLLM's internal pricing map first, then fallback to PRICING_TABLE.
    Returns per-token prices in dollars, plus per-million for readability.
    """
    try:
        info = litellm.get_model_info(model_name)
    except Exception:
        info = {}

    input_price = info.get("input_cost_per_token")
    output_price = info.get("output_cost_per_token")

    if input_price is None or output_price is None:
        fallback = PRICING_TABLE.get(model_name, {})
        input_price = fallback.get("prompt", 0.0)
        output_price = fallback.get("completion", 0.0)

    input_price = input_price or 0.0
    output_price = output_price or 0.0

    return {
        "prompt_per_token": input_price,
        "completion_per_token": output_price,
        "prompt_per_million": input_price * 1_000_000,
        "completion_per_million": output_price * 1_000_000,
    }


async def fetch_model_metadata(model_name: str) -> Dict[str, Any]:
    now = time.time()

    if model_name in MODEL_METADATA_CACHE:
        entry = MODEL_METADATA_CACHE[model_name]
        if now - entry["timestamp"] < CACHE_TTL:
            return entry["data"]

    provider_key = detect_provider(model_name)
    fallback = PROVIDER_FALLBACK.get(provider_key, {})

    try:
        info = litellm.get_model_info(model_name)
        metadata = {
            "provider": info.get("provider") or fallback.get("provider", "Unknown"),
            "context_window": info.get("max_input_tokens", "Unknown"),
            "max_output_tokens": info.get("max_output_tokens", "Unknown"),
            "supports_streaming": info.get("supports_streaming", True),
            "pricing": get_hybrid_pricing(model_name),
            "description": fallback.get("description", "No description available."),
        }
    except Exception:
        metadata = {
            "provider": fallback.get("provider", "Unknown"),
            "context_window": "Unknown",
            "max_output_tokens": "Unknown",
            "supports_streaming": False,
            "pricing": get_hybrid_pricing(model_name),
            "description": fallback.get("description", "No description available."),
        }

    MODEL_METADATA_CACHE[model_name] = {"timestamp": now, "data": metadata}
    return metadata


# MODEL PROBING sends a tiny “ping” to  model to make sure it’s alive.
# If the model responds, we test it.
# If it doesn’t, we skip it.
async def check_model_capability(model_name: str) -> bool:
    try:
        llm = ChatLiteLLM(
            model=model_name,
            streaming=False,
            callbacks=[langfuse_handler],
            metadata={"model_alias": f"{model_name}-capability"},
        )
        await llm.ainvoke([HumanMessage(content="ping")])
        return True
    except Exception:
        return False


# =========================
# TOKENIZER REGISTRY (v5.5)
# =========================

TOKENIZER_CACHE: Dict[str, Any] = {}


def get_tokenizer(model_name: str):
    """
    Provider-aware tokenizer loader.
    Uses HF tokenizers where possible, falls back to LiteLLM token_counter.
    """
    if model_name in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[model_name]

    if AutoTokenizer is None:
        TOKENIZER_CACHE[model_name] = None
        return None

    provider = detect_provider(model_name)
    tok = None

    try:
        if provider == "groq":
            # Groq Llama 3.3 70B – approximate with Meta Llama 3 70B tokenizer
            tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
        elif provider == "mistral":
            tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        elif provider == "deepseek":
            tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
        elif provider == "openai":
            # OpenAI uses tiktoken; we let LiteLLM handle it
            tok = None
        else:
            tok = None
    except Exception:
        tok = None

    TOKENIZER_CACHE[model_name] = tok
    return tok


def count_tokens(model_name: str, text: str) -> int:
    """
    Use provider-specific tokenizer if available, otherwise fallback to LiteLLM token_counter.
    """
    tok = get_tokenizer(model_name)
    if tok is not None:
        return len(tok(text)["input_ids"])
    return token_counter(model=model_name, text=text)


# =========================
# PROBE UTILITIES
# =========================

async def _probe_once(model_name: str, prompt: str) -> ProbeResult:
    llm = ChatLiteLLM(
        model=model_name,
        streaming=True,
        callbacks=[langfuse_handler],
        metadata={"model_alias": model_name},
    )

    start_time = time.perf_counter()
    ttft = 0.0
    full_content = ""

    try:
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if ttft == 0.0 and getattr(chunk, "content", None):
                ttft = (time.perf_counter() - start_time) * 1000.0 #Capture the time to first token (TTFT) in milliseconds
            if chunk.content:
                full_content += chunk.content #capture the full content for token counting and display

        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000.0

        completion_ms = max(total_latency - ttft, 0.0)
        completion_s = max(completion_ms / 1000.0, 0.001)  # epsilon fix

        completion_tokens = count_tokens(model_name, full_content)
        prompt_tokens = count_tokens(model_name, prompt)
        tps = completion_tokens / completion_s

        # Cost via hybrid pricing
        pricing = get_hybrid_pricing(model_name)
        prompt_cost = prompt_tokens * pricing["prompt_per_token"]
        completion_cost_val = completion_tokens * pricing["completion_per_token"]
        total_cost = prompt_cost + completion_cost_val

        return ProbeResult(
            status="success",
            model=model_name,
            content=full_content,
            ttft_ms=round(ttft, 2),
            total_ms=round(total_latency, 2),
            completion_ms=round(completion_ms, 2),
            tps=round(tps, 2),
            cost_usd=round(total_cost, 6),
        )
    except Exception as e:
        return ProbeResult(status="error", model=model_name, error=str(e))


@traceable(name="Probe Model (Cold+Hot)")
async def probe_model(model_name: str, prompt: str, retries: int = 1, backoff: float = 0.75) -> ProbeResult:
    if not await check_model_capability(model_name):
        return ProbeResult(
            status="error",
            model=model_name,
            error="Model capability check failed (unreachable or unsupported).",
        )

    # Cold probe
    cold_result = await _probe_once(model_name, prompt)
    if cold_result["status"] != "success":
        return cold_result

    cold_ttft = cold_result["ttft_ms"]
    cold_total = cold_result["total_ms"]

    # Hot probe with retry
    attempt = 0
    last_error: Optional[str] = None
    hot_result: Optional[ProbeResult] = None

    while attempt <= retries:
        hot_result = await _probe_once(model_name, prompt)
        if hot_result["status"] == "success":
            break
        last_error = hot_result.get("error", "Unknown error")
        attempt += 1
        if attempt <= retries:
            await asyncio.sleep(backoff * attempt)

    if hot_result is None or hot_result["status"] != "success":
        return ProbeResult(
            status="error",
            model=model_name,
            error=f"Hot probe failed after {retries + 1} attempts. Last error: {last_error}",
        )

    hot_result["cold_ttft_ms"] = cold_ttft
    hot_result["cold_total_ms"] = cold_total
    return hot_result


# =========================
# RADAR SCORE
# =========================

def compute_radar_score(result: ProbeResult, judge: JudgeResult, model_name: str, meta: Dict[str, Any]) -> float:
    ttft = result["ttft_ms"]
    tps = result["tps"]
    cost = result["cost_usd"]

    ttft_score = 1 / (1 + ttft)
    tps_score = tps / (tps + 50)
    cost_score = 1 / (1 + cost)

    confidence_score = judge["confidence"] / 10 if judge["confidence"] else 0

    ranking_score = 1.0
    if model_name in judge["ranking"]:
        idx = judge["ranking"].index(model_name)
        ranking_score = max(0.0, 1 - idx * 0.1)

    context_bonus = 0.0
    cw = meta.get("context_window")
    if isinstance(cw, (int, float)):
        context_bonus = min(cw / 100000, 1.0)

    score = (
        0.30 * ttft_score +
        0.30 * tps_score +
        0.20 * cost_score +
        0.10 * confidence_score +
        0.10 * ranking_score +
        0.05 * context_bonus
    )

    return round(score, 4)


# =========================
# JUDGE SCHEMA + PARSER
# =========================

class JudgeSchema(BaseModel):
    winner_model: Optional[str] = Field(None)
    ranking: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    confidence: int = Field(default=0)


judge_parser = PydanticOutputParser(pydantic_object=JudgeSchema)


def _build_judge_prompt(prompt: str, results: Dict[str, ProbeResult]) -> str:
    blocks = []
    for m, data in results.items():
        if data["status"] == "success":
            blocks.append(
                f"Model: {m}\n"
                f"TTFT: {data['ttft_ms']} ms\n"
                f"Total: {data['total_ms']} ms\n"
                f"TPS: {data['tps']}\n"
                f"Cost: ${data['cost_usd']}\n"
                f"Output:\n{data['content']}\n"
                "-------------------------"
            )

    comparison_text = "\n\n".join(blocks) if blocks else "No successful outputs."

    return f"""
You are an expert AI evaluator.
Return ONLY valid JSON that matches the schema.
{judge_parser.get_format_instructions()}

User prompt:
"{prompt}"

Model outputs:
{comparison_text}
"""


# =========================
# GRAPH NODES
# =========================

async def run_parallel_probes_and_metadata(state: RadarState) -> Dict:
    
    probe_tasks = [probe_model(m, state["prompt"]) for m in state["models"]] # This will fetch each model and the initial prompt given during invoke.
    meta_tasks = [fetch_model_metadata(m) for m in state["models"]]# This will fetch each model given during invoke.

    probe_results, metas = await asyncio.gather(
        asyncio.gather(*probe_tasks),
        asyncio.gather(*meta_tasks),
    )

    results = {res["model"]: res for res in probe_results}
    metadata = {m: meta for m, meta in zip(state["models"], metas)}

    return {"results": results, "metadata": metadata}


async def judge_outputs(state: RadarState) -> Dict:
    successful = [r for r in state["results"].values() if r["status"] == "success"]
    if not successful:
        return {
            "judge": JudgeResult(
                winner_model=None,
                ranking=[],
                reasoning="No successful outputs.",
                confidence=0,
            )
        }

    judge_llm = ChatLiteLLM(
        model="gpt-4o-mini",
        callbacks=[langfuse_handler],
        json_mode=True,
    )

    judge_prompt = _build_judge_prompt(state["prompt"], state["results"])
    response = await judge_llm.ainvoke([HumanMessage(content=judge_prompt)])

    # First parse attempt
    try:
        parsed: JudgeSchema = judge_parser.parse(response.content)
        ranking = parsed.ranking or sorted(list(state["results"].keys()))
        confidence = parsed.confidence or 0

        return {
            "judge": JudgeResult(
                winner_model=parsed.winner_model,
                ranking=ranking,
                reasoning=parsed.reasoning,
                confidence=confidence,
            )
        }
    except Exception:
        # Repair pass
        repair_prompt = f"""
The following output was supposed to be valid JSON matching this schema:

{judge_parser.get_format_instructions()}

But it was invalid. Fix it and return ONLY the corrected JSON.

Output:
{response.content}
"""
        repair_response = await judge_llm.ainvoke([HumanMessage(content=repair_prompt)])
        try:
            repaired = judge_parser.parse(repair_response.content)
            ranking = repaired.ranking or sorted(list(state["results"].keys()))
            confidence = repaired.confidence or 0

            return {
                "judge": JudgeResult(
                    winner_model=repaired.winner_model,
                    ranking=ranking,
                    reasoning=repaired.reasoning,
                    confidence=confidence,
                )
            }
        except Exception:
            return {
                "judge": JudgeResult(
                    winner_model=None,
                    ranking=[],
                    reasoning=f"Unparseable judge output. Original: {response.content} | Repair: {repair_response.content}",
                    confidence=0,
                )
            }


def summarize_and_score(state: RadarState) -> Dict:
    results = state["results"]
    metadata = state["metadata"]
    judge = state["judge"]

    successful = [r for r in results.values() if r["status"] == "success"]

    summary: Dict[str, Any] = {}
    scores: Dict[str, float] = {}

    if not successful:
        summary.update({"fastest": "N/A", "cheapest": "N/A", "highest_tps": "N/A"})
        return {"summary": summary, "scores": scores}

    fastest = min(successful, key=lambda x: x["ttft_ms"])
    cheapest = min(successful, key=lambda x: x["cost_usd"])
    highest_tps = max(successful, key=lambda x: x["tps"])

    summary.update(
        {
            "fastest": fastest["model"],
            "cheapest": cheapest["model"],
            "highest_tps": highest_tps["model"],
        }
    )

    for model, r in results.items():
        if r.get("status") != "success":
            continue
        meta = metadata.get(model, {})
        scores[model] = compute_radar_score(r, judge, model, meta)

    summary["scores"] = scores
    return {"summary": summary, "scores": scores}


# =========================
# GRAPH
# =========================

workflow = StateGraph(RadarState)

workflow.add_node("probe_and_metadata", run_parallel_probes_and_metadata)
workflow.add_node("judge_outputs", judge_outputs)
workflow.add_node("summarize_and_score", summarize_and_score)

workflow.set_entry_point("probe_and_metadata")
workflow.add_edge("probe_and_metadata", "judge_outputs")
workflow.add_edge("judge_outputs", "summarize_and_score")
workflow.add_edge("summarize_and_score", END)

app = workflow.compile(name="AI-Latency-Radar-v5.5")


# =========================
# EXECUTION
# =========================

async def main():
    input_state: RadarState = {
        "models": [
            "gpt-3.5-turbo",
            "groq/llama-3.3-70b-versatile",
        ],
        "prompt": "Explain atomic resonance in one sentence.",
        "results": {},
        "judge": JudgeResult(
            winner_model=None,
            ranking=[],
            reasoning="",
            confidence=0,
        ),
        "summary": {},
        "metadata": {},
        "scores": {},
    }

    print("🚀 Radar Sweep v5.5 Started...")
    final_state: RadarState = await app.ainvoke(input_state)

    results = final_state["results"]
    judge = final_state["judge"]
    summary = final_state["summary"]
    metadata = final_state["metadata"]
    scores = final_state["scores"]

    print("\n--- RADAR RESULTS ---")
    for model, metrics in results.items():
        if metrics["status"] == "success":
            print(
                f"[{model}] "
                f"Cold TTFT: {metrics['cold_ttft_ms']}ms | "
                f"Hot TTFT: {metrics['ttft_ms']}ms | "
                f"Total: {metrics['total_ms']}ms | "
                f"TPS: {metrics['tps']} | "
                f"Cost: ${metrics['cost_usd']:.6f} | "
                f"RadarScore: {scores.get(model, 0)}"
            )
        else:
            print(f"[{model}] FAILED -> {metrics.get('error')}")

    print("\n--- SUMMARY ---")
    print(summary)

    print("\n--- METADATA ---")
    for m, meta in metadata.items():
        print(f"{m}: {meta}")

    print("\n--- JUDGE ---")
    print(judge)


if __name__ == "__main__":
    asyncio.run(main())
