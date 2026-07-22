"""Hermetic coverage for configurable English smoke evaluation parsing."""

from scripts.eval import run_eval


def test_english_smoke_question_file_parses_all_subjects():
    records = run_eval.parse(
        (run_eval.REPO / "eval_questions_en_smoke.md").read_text(encoding="utf-8")
    )

    assert len(records) == 6
    assert {record["subject"] for record in records} == {
        "physics",
        "chemistry",
        "biology",
    }
    assert records[0]["question"].startswith("What is the boiling point")
    assert records[-1]["expected"].startswith("I can only answer")
