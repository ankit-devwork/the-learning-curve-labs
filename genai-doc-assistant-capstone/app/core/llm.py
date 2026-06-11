from litellm import acompletion
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id
from pycorekit.logging.logger import get_logger
from app.core.settings import settings
log = get_logger("llm")
MODEL = settings.models.llm_model

async def llm(prompt: str) -> str:
    cid = get_current_correlation_id() or "unknown"
    bound = log.bind(correlation_id=cid, llm_model=MODEL)

    with start_trace("llm_call", inputs={"prompt": prompt}) as obs:
        lf_obs = obs["langfuse"]
        ls_run = obs["langsmith"]

        try:
            response = await acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )

            answer = response["choices"][0]["message"]["content"]

            # Safe Langfuse update
            if lf_obs:
                try:
                    lf_obs.update(output={"answer": answer})
                except Exception as e:
                    bound.warning("Langfuse update failed", error=str(e))

            # Safe LangSmith update
            if ls_run:
                try:
                    ls_run.update(outputs={"answer": answer})
                except Exception as e:
                    bound.warning("LangSmith update failed", error=str(e))

            return answer

        except Exception as e:
            bound.exception("LLM call failed", error=str(e))

            # Safe Langfuse error update
            if lf_obs:
                try:
                    lf_obs.update(output={"error": str(e)})
                except Exception:
                    pass

            # Safe LangSmith error update
            if ls_run:
                try:
                    ls_run.update(error=str(e))
                except Exception:
                    pass

            raise
