import pytest

from app.services.export_utils import chart_rows_to_csv


def test_chart_rows_to_csv():
    csv_text = chart_rows_to_csv(
        title="Sales by region",
        labels=["North", "South"],
        values=[100, 200],
    )
    assert "label,value" in csv_text
    assert "North,100" in csv_text
    assert "South,200" in csv_text


def test_weak_concepts_hash_stable():
    from app.services.quiz_service import _weak_concepts_hash

    assert _weak_concepts_hash(["b", "a"]) == _weak_concepts_hash(["a", "b"])
