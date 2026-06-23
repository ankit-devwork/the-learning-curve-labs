"""Client-side export helpers mirrored for tests and optional server export."""


def chart_rows_to_csv(*, title: str, labels: list[str], values: list[float | int]) -> str:
    lines = ["label,value"]
    for label, value in zip(labels, values, strict=False):
        safe_label = str(label).replace('"', '""')
        if "," in safe_label or "\n" in safe_label:
            safe_label = f'"{safe_label}"'
        lines.append(f"{safe_label},{value}")
    _ = title
    return "\n".join(lines)
