import asyncio
import io

import pandas as pd
import pytest

from app.core.resilience import CircuitBreaker, CircuitState
from app.services.excel_charts import build_charts_from_plan, parse_chart_plan
from app.services.excel_profiling import profile_dataframe, read_dataframe


def test_read_and_profile_csv():
    csv = "region,sales\nNorth,100\nSouth,80\nEast,120\n".encode()
    df = read_dataframe(csv, "sales.csv")
    profile = profile_dataframe(df)
    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert profile["columns"][0]["name"] == "region"


def test_parse_chart_plan_json():
    raw = """
    {
      "charts": [
        {
          "id": "c1",
          "title": "Sales by region",
          "chart_type": "bar",
          "x_column": "region",
          "y_column": "sales",
          "aggregation": "sum"
        }
      ]
    }
    """
    items = parse_chart_plan(raw, max_charts=3)
    assert len(items) == 1
    assert items[0]["chart_type"] == "bar"


def test_build_charts_from_plan():
    df = pd.read_csv(io.StringIO("region,sales\nNorth,100\nSouth,80\nEast,120\n"))
    plan = [
        {
            "id": "c1",
            "title": "Sales by region",
            "chart_type": "bar",
            "x_column": "region",
            "y_column": "sales",
            "aggregation": "sum",
        }
    ]
    charts = build_charts_from_plan(df, plan)
    assert len(charts) == 1
    assert charts[0]["labels"] == ["East", "North", "South"]
    assert charts[0]["values"][0] == 120.0


def test_circuit_breaker_opens_after_failures():
    breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_seconds=60)

    async def fail():
        raise RuntimeError("connection timeout")

    async def run():
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
        assert breaker.state == CircuitState.CLOSED

        with pytest.raises(RuntimeError):
            await breaker.call(fail)
        assert breaker.state == CircuitState.OPEN

        with pytest.raises(Exception, match="circuit breaker is open"):
            await breaker.call(fail)

    asyncio.run(run())
