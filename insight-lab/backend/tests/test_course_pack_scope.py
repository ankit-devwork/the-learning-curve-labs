"""Course pack helper tests."""

from app.services.course_pack_utils import sample_homework_question


def test_sample_homework_question_prefers_study_guide_sample():
    question = sample_homework_question(
        {
            "title": "Ratios",
            "content": {"sample_questions": ["What is a ratio?", "How do you simplify?"]},
        }
    )
    assert question == "What is a ratio?"


def test_sample_homework_question_falls_back_to_default():
    assert sample_homework_question(None) == "Explain the key concepts from this document step by step."
