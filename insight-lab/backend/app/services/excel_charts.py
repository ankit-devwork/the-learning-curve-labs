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


class CustomChartRequest(BaseModel):
    chart_type: str = Field(pattern=r"^(bar|line|pie|scatter)$")
    x_column: str
    y_column: str | None = None
    aggregation: str = Field(default="sum", pattern=r"^(sum|mean|count|none)$")
    title: str | None = None

    @field_validator("chart_type", "aggregation", mode="before")
    @classmethod
    def normalize_lowercase(cls, value: object) -> object:
        return str(value).lower().strip() if value is not None else value


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


def _sort_chart_stats(stats: pd.Series, chart_type: str) -> pd.Series:
    if chart_type == "line":
        parsed_dates = pd.to_datetime(stats.index, errors="coerce")
        if parsed_dates.notna().all():
            ordered = stats.copy()
            ordered.index = parsed_dates
            return ordered.sort_index().head(12)
        return stats.sort_index().head(12)
    return stats.sort_values(ascending=False).head(12)


def _format_chart_label(index: object) -> str:
    if isinstance(index, pd.Timestamp):
        return index.strftime("%Y-%m-%d")
    return str(index)


def _to_chart_floats(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64").astype(float) / 1_000_000_000.0
    if pd.api.types.is_timedelta64_dtype(series):
        return series.astype("int64").astype(float)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).astype(float)
    return pd.to_numeric(series, errors="coerce")


def _stat_value_to_float(value: object) -> float:
    if isinstance(value, pd.Timestamp):
        return round(float(value.timestamp()), 4)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    return round(float(value), 4)


def _build_scatter_data(working: pd.DataFrame, x_col: str, y_col: str) -> tuple[list[str], list[float]]:
    x_numeric = _to_chart_floats(working[x_col])
    y_numeric = _to_chart_floats(working[y_col])
    mask = x_numeric.notna() & y_numeric.notna()
    filtered = working.loc[mask].head(100)
    if filtered.empty:
        return [], []

    x_values = _to_chart_floats(filtered[x_col])
    y_values = _to_chart_floats(filtered[y_col])
    labels = [str(round(float(value), 4)) for value in x_values.tolist()]
    values = [round(float(value), 4) for value in y_values.tolist()]
    return labels, values


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
            labels, values = _build_scatter_data(working, x_col, y_col)
        elif y_col and chart_type in {"bar", "line", "pie"}:
            if aggregation == "count":
                stats = working.groupby(x_col, dropna=True)[y_col].count()
            else:
                numeric_y = _to_chart_floats(working[y_col])
                grouped = working.assign(_chart_y=numeric_y).groupby(x_col, dropna=True)["_chart_y"]
                if aggregation == "mean":
                    stats = grouped.mean()
                else:
                    stats = grouped.sum()
            stats = _sort_chart_stats(stats, chart_type)
            labels = [_format_chart_label(index) for index in stats.index.tolist()]
            values = [_stat_value_to_float(value) for value in stats.tolist()]
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


def _default_chart_title(request: CustomChartRequest) -> str:
    if request.y_column:
        agg = "" if request.aggregation == "sum" else f" ({request.aggregation})"
        return f"{request.y_column}{agg} by {request.x_column}"
    return f"{request.x_column} distribution"


def build_custom_chart(df: pd.DataFrame, request: CustomChartRequest) -> dict[str, Any]:
    if request.chart_type in {"bar", "line", "scatter"} and not request.y_column:
        raise ValueError(f"{request.chart_type} charts require a Y column")

    if request.x_column not in df.columns:
        raise ValueError(f"Column not found: {request.x_column}")
    if request.y_column and request.y_column not in df.columns:
        raise ValueError(f"Column not found: {request.y_column}")

    plan_item = {
        "id": "custom",
        "title": request.title or _default_chart_title(request),
        "chart_type": request.chart_type,
        "x_column": request.x_column,
        "y_column": request.y_column,
        "aggregation": request.aggregation,
    }
    charts = build_charts_from_plan(df, [plan_item])
    if not charts:
        raise ValueError("Could not build chart with the selected columns")
    chart = charts[0]
    chart["id"] = f"custom-{request.chart_type}-{request.x_column}-{request.y_column or 'count'}"
    chart["custom"] = True
    return chart
