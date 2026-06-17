import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class ChartPlanItem(BaseModel):
    id: str
    title: str
    chart_type: str = Field(pattern=r"^(bar|line|pie|scatter)$")
    x_column: str
    y_column: str | None = None
    aggregation: str = Field(default="sum", pattern=r"^(sum|mean|count|none)$")

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_string(cls, value: object) -> str:
        return str(value)

    @field_validator("chart_type", "aggregation", mode="before")
    @classmethod
    def normalize_lowercase(cls, value: object) -> object:
        return str(value).lower().strip() if value is not None else value


class ChartPlan(BaseModel):
    charts: list[ChartPlanItem]


def parse_chart_plan(raw: str, *, max_charts: int) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    payload = json.loads(text)
    if isinstance(payload, dict) and "charts" in payload:
        plan = ChartPlan.model_validate(payload)
    elif isinstance(payload, list):
        plan = ChartPlan.model_validate({"charts": payload})
    else:
        raise ValueError("Chart plan must be an object with 'charts' or a list")
    return [item.model_dump() for item in plan.charts[:max_charts]]


def _aggregate(series: pd.Series, aggregation: str) -> float:
    if aggregation == "count":
        return float(series.count())
    if aggregation == "mean":
        return float(series.mean())
    if aggregation == "sum":
        return float(series.sum())
    return float(series.iloc[0]) if len(series) else 0.0


def build_charts_from_plan(df: pd.DataFrame, plan_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for item in plan_items:
        x_col = item["x_column"]
        y_col = item.get("y_column")
        chart_type = item["chart_type"]
        aggregation = item.get("aggregation", "sum")

        if x_col not in df.columns:
            continue
        if y_col and y_col not in df.columns:
            continue

        working = df[[c for c in {x_col, y_col} if c]].dropna()
        if working.empty:
            continue

        labels: list[str]
        values: list[float]

        if chart_type == "scatter" and y_col:
            labels = working[x_col].astype(str).head(100).tolist()
            values = working[y_col].astype(float).head(100).tolist()
        elif y_col and chart_type in {"bar", "line", "pie"}:
            grouped = working.groupby(x_col, dropna=True)[y_col]
            if aggregation == "count":
                stats = grouped.count()
            elif aggregation == "mean":
                stats = grouped.mean()
            else:
                stats = grouped.sum()
            stats = stats.sort_values(ascending=False).head(12)
            labels = [str(index) for index in stats.index.tolist()]
            values = [round(float(value), 4) for value in stats.tolist()]
        else:
            counts = working[x_col].astype(str).value_counts().head(12)
            labels = counts.index.tolist()
            values = [float(v) for v in counts.tolist()]

        if not labels:
            continue

        charts.append(
            {
                "id": item["id"],
                "title": item["title"],
                "chart_type": chart_type,
                "x_column": x_col,
                "y_column": y_col,
                "aggregation": aggregation,
                "labels": labels,
                "values": values,
            }
        )
    return charts
