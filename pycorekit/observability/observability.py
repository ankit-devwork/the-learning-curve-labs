"""
High-level observability helpers.

All functions:
- Use unified start_trace()
- Never break if LangSmith or Langfuse fail
- Always update spans safely
- Always return or re-raise the original error (never masking)
"""

from typing import Any, Callable, Awaitable

from pycorekit.core_logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id
from pycorekit.tracing.tracing import start_trace, get_langsmith_client


def _update_langsmith(ls_run, langsmith, **kwargs) -> None:
    if ls_run and langsmith:
        try:
            langsmith.update_run(ls_run["id"], **kwargs)
        except Exception as e:
            logger.warning("LangSmith update failed", error=str(e))


async def observe_llm(name: str, func: Callable[..., Awaitable[str]], *, prompt: str) -> str:
    cid = get_current_correlation_id() or "unknown"
    bound = logger.bind(correlation_id=cid, llm_call=name)
    langsmith = get_langsmith_client()

    bound.info("LLM call started", prompt_preview=prompt[:80])

    with start_trace(name, inputs={"prompt": prompt}) as obs:
        lf_obs = obs["langfuse"]
        ls_run = obs["langsmith"]

        try:
            answer = await func(prompt)

            if lf_obs:
                try:
                    lf_obs.update(output={"answer": answer})
                except Exception as e:
                    bound.warning("Langfuse update failed", error=str(e))

            _update_langsmith(ls_run, langsmith, outputs={"answer": answer})
            bound.info("LLM call completed", answer_preview=answer[:80])
            return answer

        except Exception as e:
            bound.exception("LLM call failed", error=str(e))
            if lf_obs:
                try:
                    lf_obs.update(output={"error": str(e)})
                except Exception:
                    pass
            _update_langsmith(ls_run, langsmith, error=str(e))
            raise


async def observe_rag(
    name: str,
    retriever: Callable[[str], Awaitable[list]],
    llm: Callable[[str], Awaitable[str]],
    *,
    query: str,
) -> dict:
    cid = get_current_correlation_id() or "unknown"
    bound = logger.bind(correlation_id=cid, rag=name)
    langsmith = get_langsmith_client()

    bound.info("RAG pipeline started", query=query)

    with start_trace(name, inputs={"query": query}) as obs:
        lf_obs = obs["langfuse"]
        ls_run = obs["langsmith"]

        try:
            docs = await retriever(query)
            context = "\n".join([getattr(d, "page_content", str(d)) for d in docs])
            answer = await llm(f"Context:\n{context}\n\nQuestion: {query}")
            result = {"answer": answer, "documents": docs}

            if lf_obs:
                try:
                    lf_obs.update(output=result)
                except Exception as e:
                    bound.warning("Langfuse update failed", error=str(e))

            _update_langsmith(ls_run, langsmith, outputs=result)
            bound.info("RAG pipeline completed")
            return result

        except Exception as e:
            bound.exception("RAG pipeline failed", error=str(e))
            if lf_obs:
                try:
                    lf_obs.update(output={"error": str(e)})
                except Exception:
                    pass
            _update_langsmith(ls_run, langsmith, error=str(e))
            raise


async def observe_agent_step(name: str, func: Callable[..., Awaitable[Any]], *, step_input: dict) -> Any:
    cid = get_current_correlation_id() or "unknown"
    bound = logger.bind(correlation_id=cid, agent_step=name)
    langsmith = get_langsmith_client()

    bound.info("Agent step started", step_input=step_input)

    with start_trace(name, inputs=step_input) as obs:
        lf_obs = obs["langfuse"]
        ls_run = obs["langsmith"]

        try:
            result = await func(**step_input)

            if lf_obs:
                try:
                    lf_obs.update(output=result)
                except Exception as e:
                    bound.warning("Langfuse update failed", error=str(e))

            _update_langsmith(ls_run, langsmith, outputs=result)
            bound.info("Agent step completed")
            return result

        except Exception as e:
            bound.exception("Agent step failed", error=str(e))
            if lf_obs:
                try:
                    lf_obs.update(output={"error": str(e)})
                except Exception:
                    pass
            _update_langsmith(ls_run, langsmith, error=str(e))
            raise


async def observe_http(name: str, func: Callable[..., Awaitable[Any]], *, url: str, method: str = "GET") -> Any:
    cid = get_current_correlation_id() or "unknown"
    bound = logger.bind(correlation_id=cid, http_call=name)
    langsmith = get_langsmith_client()

    bound.info("HTTP call started", url=url, method=method)

    with start_trace(name, inputs={"url": url, "method": method}) as obs:
        lf_obs = obs["langfuse"]
        ls_run = obs["langsmith"]

        try:
            result = await func()

            if lf_obs:
                try:
                    lf_obs.update(output=result)
                except Exception as e:
                    bound.warning("Langfuse update failed", error=str(e))

            _update_langsmith(ls_run, langsmith, outputs=result)
            bound.info("HTTP call completed")
            return result

        except Exception as e:
            bound.exception("HTTP call failed", error=str(e))
            if lf_obs:
                try:
                    lf_obs.update(output={"error": str(e)})
                except Exception:
                    pass
            _update_langsmith(ls_run, langsmith, error=str(e))
            raise


async def observe_db(name: str, func: Callable[..., Awaitable[Any]], *, query: str) -> Any:
    cid = get_current_correlation_id() or "unknown"
    bound = logger.bind(correlation_id=cid, db_query=name)
    langsmith = get_langsmith_client()

    bound.info("DB query started", query=query)

    with start_trace(name, inputs={"query": query}) as obs:
        lf_obs = obs["langfuse"]
        ls_run = obs["langsmith"]

        try:
            result = await func()

            if lf_obs:
                try:
                    lf_obs.update(output=result)
                except Exception as e:
                    bound.warning("Langfuse update failed", error=str(e))

            _update_langsmith(ls_run, langsmith, outputs=result)
            bound.info("DB query completed")
            return result

        except Exception as e:
            bound.exception("DB query failed", error=str(e))
            if lf_obs:
                try:
                    lf_obs.update(output={"error": str(e)})
                except Exception:
                    pass
            _update_langsmith(ls_run, langsmith, error=str(e))
            raise
