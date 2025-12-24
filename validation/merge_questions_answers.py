"""
Utility script to merge Question.csv and Answer.csv for inspection.
Rows align to Question.csv; choices are emitted as a JSON string of {value, isCorrect}.

Configure paths in main() call below. No arg parsing is used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def merge_question_answer(question_path: Path, answer_path: Path, output_path: Path) -> None:
    answer_lookup: Dict[str, List[Dict[str, str | bool]]] = {}

    with answer_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qid = row.get("questionId")
            if not qid:
                continue
            value = row.get("value")
            is_correct_raw = row.get("isCorrect")
            is_correct = str(is_correct_raw).lower() == "true"
            answer_lookup.setdefault(qid, []).append({"value": value, "isCorrect": is_correct})

    with question_path.open(newline="", encoding="utf-8") as qh, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as out:
        q_reader = csv.DictReader(qh)
        fieldnames = list(q_reader.fieldnames or [])
        if "choices" not in fieldnames:
            fieldnames.append("choices")
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in q_reader:
            qid = row.get("id")
            choices_dict = {c["value"]: c["isCorrect"] for c in answer_lookup.get(qid, [])}
            row = dict(row)
            row["choices"] = json.dumps(choices_dict, ensure_ascii=False)
            writer.writerow(row)

    print(f"Wrote merged question+answer file to {output_path}")


def main() -> None:
    base = Path("_data/exam_result")
    out_dir = Path("validation/_data_for_validation")
    out_dir.mkdir(exist_ok=True)
    merge_question_answer(
        question_path=base / "Question.csv",
        answer_path=base / "Answer.csv",
        output_path=out_dir / "questions_with_choices.csv",
    )


if __name__ == "__main__":
    main()
