"""
Utility script to stitch ExamResult, ExamQuestionResult, ExamAnswerResult, Question, and Answer
into a single CSV for validation. Rows align to ExamQuestionResult entries.

Configure paths in main() call below. No arg parsing is used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def merge_exam_results(
    exam_result_path: Path,
    exam_question_result_path: Path,
    exam_answer_result_path: Path,
    question_path: Path,
    answer_path: Path,
    output_path: Path,
) -> None:
    exam_results = _load_table(exam_result_path, key="id")
    questions = _load_table(question_path, key="id")
    answer_lookup = _load_answers(answer_path)
    answers_by_question_result = _group_answers_by_exam_question(exam_answer_result_path)

    with exam_question_result_path.open(newline="", encoding="utf-8") as fh, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(fh)
        fieldnames = [
            "examResultQuestionId",
            "examResultId",
            "userId",
            "testId",
            "testTitle",
            "attemptNumber",
            "questionId",
            "questionText",
            "studentAnswers",
            "studentAnswersWithCorrectness",
            "correctAnswers",
            "allChoices",
            "answeredCorrectly",
            "createdAt",
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            exam_result_id = row.get("examResultId")
            question_id = row.get("questionId")
            exam_res = exam_results.get(exam_result_id, {})
            question = questions.get(question_id, {})
            answers = answers_by_question_result.get(row.get("id"), [])

            student_answers = [a.get("answerValue") for a in answers]
            student_answers_with_correctness = [
                {"value": a.get("answerValue"), "isCorrect": _to_bool(a.get("isCorrect"))}
                for a in answers
            ]
            correct_answers = [
                a["value"] for a in answer_lookup.get(question_id, []) if a.get("isCorrect")
            ]
            all_choices = {a["value"]: a.get("isCorrect") for a in answer_lookup.get(question_id, [])}
            answered_correctly = any(_to_bool(a.get("isCorrect")) for a in answers)

            out_row = {
                "examResultQuestionId": row.get("id"),
                "examResultId": exam_result_id,
                "userId": exam_res.get("userId"),
                "testId": exam_res.get("testId"),
                "testTitle": exam_res.get("testTitle"),
                "attemptNumber": exam_res.get("attemptNumber"),
                "questionId": question_id,
                "questionText": question.get("question"),
                "studentAnswers": json.dumps(student_answers, ensure_ascii=False),
                "studentAnswersWithCorrectness": json.dumps(
                    student_answers_with_correctness, ensure_ascii=False
                ),
                "correctAnswers": json.dumps(correct_answers, ensure_ascii=False),
                "allChoices": json.dumps(all_choices, ensure_ascii=False),
                "answeredCorrectly": answered_correctly,
                "createdAt": row.get("createdAt"),
            }
            writer.writerow(out_row)

    print(f"Wrote merged exam results file to {output_path}")


def _load_table(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            k = row.get(key)
            if k:
                data[k] = row
    return data


def _load_answers(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qid = row.get("questionId")
            if not qid:
                continue
            lookup.setdefault(qid, []).append(
                {
                    "value": row.get("value"),
                    "isCorrect": _to_bool(row.get("isCorrect")),
                }
            )
    return lookup


def _group_answers_by_exam_question(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            eq_id = row.get("examResultQuestionId")
            if not eq_id:
                continue
            grouped.setdefault(eq_id, []).append(row)
    return grouped


def _to_bool(val: Any) -> bool:
    return str(val).lower() == "true"


def main() -> None:
    base = Path("_data/exam_result")
    out_dir = Path("validation/_data_for_validation")
    out_dir.mkdir(exist_ok=True)
    merge_exam_results(
        exam_result_path=base / "ExamResult.csv",
        exam_question_result_path=base / "ExamQuestionResult.csv",
        exam_answer_result_path=base / "ExamAnswerResult.csv",
        question_path=base / "Question.csv",
        answer_path=base / "Answer.csv",
        output_path=out_dir / "exam_result_flat.csv",
    )


if __name__ == "__main__":
    main()
