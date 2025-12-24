"""
Utility script to group questions by domain per test_id.
Outputs a CSV with columns: testId, domain, questionCount.
Configure paths in main(); no arg parsing.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def questions_by_domain(question_path: Path, output_path: Path) -> None:
    counts = defaultdict(lambda: defaultdict(int))

    with question_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            test_id = row.get("testId") or ""
            domain = row.get("domain") or "Unknown"
            counts[test_id][domain] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as out:
        fieldnames = ["testId", "domain", "questionCount"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for test_id, domains in counts.items():
            for domain, cnt in domains.items():
                writer.writerow(
                    {
                        "testId": test_id,
                        "domain": domain,
                        "questionCount": cnt,
                    }
                )

    print(f"Wrote questions-by-domain report to {output_path}")


def main() -> None:
    base = Path("_data/exam_result")
    out_dir = Path("validation/_data_for_validation")
    questions_by_domain(
        question_path=base / "Question.csv",
        output_path=out_dir / "questions_by_domain.csv",
    )


if __name__ == "__main__":
    main()
