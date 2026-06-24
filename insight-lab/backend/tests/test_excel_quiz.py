from app.services.quiz_questions import QuizDraft, QuizQuestionDraft, excel_draft_to_rows


def test_excel_draft_to_rows_maps_chart_source():
    draft = QuizDraft(
        title="Sales quiz",
        questions=[
            QuizQuestionDraft(
                question_text="Which region had the highest sales?",
                options=["North", "South"],
                correct_option_index=0,
                explanation="North leads in Q4.",
                source_chart_index=1,
            )
        ],
    )
    rows = excel_draft_to_rows(draft, chart_ids=["chart-a", "chart-b"])
    assert rows[0]["source_chunk_id"] == "chart:chart-b"
