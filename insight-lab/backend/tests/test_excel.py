import asyncio
import io

import pandas as pd
import pytest

from app.core.resilience import CircuitBreaker, CircuitState
from app.services.excel_charts import (
    CustomChartRequest,
    build_charts_from_plan,
    build_custom_chart,
    parse_chart_plan,
)
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


def test_parse_chart_plan_coerces_numeric_ids():
    raw = """
    {
      "charts": [
        {
          "id": 1,
          "title": "Sales by region",
          "chart_type": "BAR",
          "x_column": "region",
          "y_column": "sales",
          "aggregation": "SUM"
        }
      ]
    }
    """
    items = parse_chart_plan(raw, max_charts=3)
    assert items[0]["id"] == "1"
    assert items[0]["chart_type"] == "bar"
    assert items[0]["aggregation"] == "sum"


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


def test_build_line_chart_sorts_by_date():
    df = pd.read_csv(
        io.StringIO(
            "Date,Quantity\n"
            "2025-05-11,86\n"
            "2024-03-21,63\n"
            "2024-11-02,75\n"
        )
    )
    plan = [
        {
            "id": "c2",
            "title": "Quantity by date",
            "chart_type": "line",
            "x_column": "Date",
            "y_column": "Quantity",
            "aggregation": "sum",
        }
    ]
    charts = build_charts_from_plan(df, plan)
    assert charts[0]["labels"] == ["2024-03-21", "2024-11-02", "2025-05-11"]
    assert charts[0]["values"] == [63.0, 75.0, 86.0]


def test_build_custom_chart_bar():
    df = pd.read_csv(io.StringIO("region,sales\nNorth,100\nSouth,80\nEast,120\n"))
    request = CustomChartRequest(
        chart_type="bar",
        x_column="region",
        y_column="sales",
        aggregation="sum",
        title="Regional sales",
    )
    chart = build_custom_chart(df, request)
    assert chart["custom"] is True
    assert chart["title"] == "Regional sales"
    assert chart["chart_type"] == "bar"
    assert len(chart["labels"]) == 3


def test_build_custom_chart_requires_y_for_scatter():
    df = pd.read_csv(io.StringIO("region,sales\nNorth,100\n"))
    request = CustomChartRequest(chart_type="scatter", x_column="region")
    with pytest.raises(ValueError, match="require a Y column"):
        build_custom_chart(df, request)


def test_build_custom_pie_without_y_column():
    df = pd.read_csv(io.StringIO("product,qty\nChair,3\nDesk,2\nChair,1\n"))
    request = CustomChartRequest(chart_type="pie", x_column="product")
    chart = build_custom_chart(df, request)
    assert chart["chart_type"] == "pie"
    assert "Chair" in chart["labels"]


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
