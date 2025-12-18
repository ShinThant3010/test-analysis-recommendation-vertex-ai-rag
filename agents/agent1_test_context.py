# agents/agent1_test_context.py
from typing import Dict, Any
import pandas as pd

from config import TEST_RESULT_PATH

# Columns for Agent 1 output
CORE_COLS = [
    "id",
    "examContentId",
    "userId",
    "attemptNumber",
    "totalAttempts",
    "durationTakenMs",  
    "earnedScore",
    "totalScore",
    "status",
    "createdAt",
]


def _serialize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert non-JSON types (e.g., Timestamp) to safe values."""
    serialized: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


def get_student_test_history(
    test_id: int,
    student_id: int,
    csv_path: str = TEST_RESULT_PATH,
) -> Dict[str, Any]:
    """
    Core Agent 1 – Test Context & Validation: Test Filtering (CSV version)

    Output:
        {
          "status": "ok" | "no_current_test" | "no_tests_for_student",
          "input": {...},
          "current_test_result": {...} | None,
          "history_test_result": [...] | None,
          "notes": [...],
        }
    """
    result: Dict[str, Any] = {
        "status": "ok",
        "input": {"test_id": test_id, "student_id": student_id},
        "current_test_result": None,
        "history_test_result": None,
        "notes": [],
    }

    df = pd.read_csv(csv_path)

    # Parse timestamp
    df["testTakenDT"] = pd.to_datetime(df["createdAt"])

    # Filter by student
    df_student = df[df["userId"] == student_id].copy()
    if df_student.empty:
        result["status"] = "no_tests_for_student"
        result["notes"].append("This student has not taken any tests.")
        return result

    # Filter by student + test
    # df_student_test = df_student[df_student["id"] == test_id].copy()
    df_student_test = df_student[df_student["examContentId"] == test_id].copy()
    if df_student_test.empty:
        result["status"] = "no_current_test"
        result["notes"].append("Student has test history, but not for this test_id.")
        return result

    # Latest attempt = current test
    df_student_test_sorted = df_student_test.sort_values(
        ["attemptNumber", "testTakenDT"], ascending=[False, False]
    )
    current_row = df_student_test_sorted.iloc[0]
    result["current_test_result"] = _serialize_record(
        current_row[CORE_COLS].to_dict()
    )

    # one previous attempts = history

    history_row = df_student_test_sorted.iloc[1:2]
    result["history_test_result"] = _serialize_record(
        history_row[CORE_COLS].to_dict()
    )

    if not result["history_test_result"]:
        result["notes"].append(
            "No previous attempts for this test_id (first time taking this test)."
        )

    return result
