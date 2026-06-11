from pathlib import Path
import json
import yaml
import pandas as pd
from pdfminer.high_level import extract_text as extract_pdf_text


def parse_file_content(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")

    if ext == "pdf":
        return extract_pdf_text(str(path))

    if ext == "txt":
        return path.read_text(errors="ignore")

    if ext == "csv":
        df = pd.read_csv(path)
        return df.to_string()

    if ext == "xlsx":
        df = pd.read_excel(path)
        return df.to_string()

    if ext == "json":
        data = json.loads(path.read_text())
        return json.dumps(data, indent=2)

    if ext in ("yaml", "yml"):
        data = yaml.safe_load(path.read_text())
        return yaml.dump(data)

    raise ValueError(f"Unsupported extension: {ext}")
