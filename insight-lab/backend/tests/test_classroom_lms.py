from app.services.export_utils import build_imsmanifest_xml, quiz_to_qti_xml


def test_build_imsmanifest_xml_includes_resources():
    xml = build_imsmanifest_xml(
        title="Biology 101",
        resources=[
            {
                "identifier": "res1",
                "title": "Chapter 1 Quiz",
                "type": "imsqti_xmlv1p2",
                "href": "assessments/ch1-quiz.xml",
            }
        ],
    )
    assert "Biology 101" in xml
    assert "<manifest" in xml
    assert "assessments/ch1-quiz.xml" in xml
    assert "imsqti_xmlv1p2" in xml


def test_quiz_to_qti_xml_still_valid():
    xml = quiz_to_qti_xml(
        title="Sample",
        questions=[
            {
                "question_text": "Test?",
                "options": ["A", "B"],
                "correct_option_index": 0,
            }
        ],
    )
    assert "questestinterop" in xml
