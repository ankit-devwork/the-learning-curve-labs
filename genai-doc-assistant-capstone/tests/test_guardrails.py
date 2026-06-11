from app.core.guardrails import apply_guardrails, detect_unsafe_ops, detect_unsafe_content


def test_detect_unsafe_ops():
    assert detect_unsafe_ops("please run rm -rf /") is True
    assert detect_unsafe_ops("summarize this document") is False


def test_detect_unsafe_content():
    assert detect_unsafe_content("how to hack a server") is True
    assert detect_unsafe_content("what are the project deadlines") is False


def test_apply_guardrails_blocks_unsafe_answer():
    answer = apply_guardrails("run rm -rf on the server")
    assert "cannot assist" in answer.lower()
