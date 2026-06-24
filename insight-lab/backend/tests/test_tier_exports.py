from app.services.export_utils import course_pack_to_markdown, quiz_to_qti_xml


def test_quiz_to_qti_xml_includes_questions():
    xml = quiz_to_qti_xml(
        title="Sample Quiz",
        questions=[
            {
                "question_text": "What is 2+2?",
                "options": ["3", "4", "5"],
                "correct_option_index": 1,
            }
        ],
    )
    assert "Sample Quiz" in xml
    assert "What is 2+2?" in xml
    assert "questestinterop" in xml
    assert 'ident="B"' in xml or "4" in xml


def test_course_pack_to_markdown_includes_summaries():
    markdown = course_pack_to_markdown(
        workspace_name="Biology 101",
        documents=[
            {"filename": "chapter1.pdf", "summary": "Cells are the basic unit of life."},
            {"filename": "chapter2.pdf", "summary": "Photosynthesis converts light to energy."},
        ],
    )
    assert "# Course pack — Biology 101" in markdown
    assert "## chapter1.pdf" in markdown
    assert "Cells are the basic unit of life." in markdown
    assert "Photosynthesis converts light to energy." in markdown


def test_course_pack_to_markdown_labels_spreadsheets():
    markdown = course_pack_to_markdown(
        workspace_name="Lab data",
        documents=[
            {"filename": "results.xlsx", "summary": "Sales grew 12% quarter over quarter.", "file_type": "excel"},
        ],
    )
    assert "## results.xlsx (spreadsheet)" in markdown
    assert "Sales grew 12%" in markdown
