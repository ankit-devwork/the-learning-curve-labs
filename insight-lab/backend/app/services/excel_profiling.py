import io
from typing import Any

import pandas as pd
from pycorekit.exceptions.file import FileException

from app.core.yaml_config import get_yaml_config


def _extension_for_filename(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def read_dataframe(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = _extension_for_filename(filename)
    buffer = io.BytesIO(raw_bytes)
    try:
        if ext == "csv":
            df = pd.read_csv(buffer)
        elif ext in {"xlsx", "xls"}:
            df = pd.read_excel(buffer)
        else:
            raise FileException(f"Unsupported Excel extension: .{ext}")
    except FileException:
        raise
    except Exception as exc:
        raise FileException(f"Failed to parse spreadsheet: {exc}", status_code=400) from exc

    cfg = get_yaml_config().excel
    if len(df) > cfg.max_rows:
        df = df.head(cfg.max_rows)
    if len(df.columns) > cfg.max_columns:
        df = df.iloc[:, : cfg.max_columns]
    if df.empty:
        raise FileException("Spreadsheet has no rows", status_code=400)
    return df


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "text"


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    cfg = get_yaml_config().excel
    columns: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        kind = _column_kind(series)
        non_null = series.dropna()
        column: dict[str, Any] = {
            "name": str(name),
            "dtype": kind,
            "null_pct": round(float(series.isna().mean()), 4),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if kind == "numeric" and not non_null.empty:
            column["min"] = float(non_null.min())
            column["max"] = float(non_null.max())
            column["mean"] = round(float(non_null.mean()), 4)
        samples = non_null.head(5).astype(str).tolist()
        column["sample_values"] = samples
        columns.append(column)

    sample_rows = (
        df.head(cfg.sample_rows_for_llm)
        .astype(str)
        .fillna("")
        .to_dict(orient="records")
    )
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "sample_rows": sample_rows,
    }
