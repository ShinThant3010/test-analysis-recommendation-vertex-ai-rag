# agents/agent2_incorrect_questions.py
from typing import Dict, Any, List
import pandas as pd

from config import (
    QUESTION_PATH,
    ANSWER_PATH,
    TQ_PATH,
    TA_PATH,
)


def get_incorrect_question_cases(
    agent1_output: Dict[str, Any],      # the output from Agent 1
    question_path: str = QUESTION_PATH, # to get question text and metadata
    answer_path: str = ANSWER_PATH,     # to get correct answers
    tq_path: str = TQ_PATH,             # to get test question results
    ta_path: str = TA_PATH,             # to get test answer results
) -> Dict[str, Any]:
    """
    Core Agent 2 – Extract questions with incorrect answers for the *current test*.
    """

    test_id = agent1_output.get("input", {}).get("test_id")
    student_id = agent1_output.get("input", {}).get("student_id")

    result: Dict[str, Any] = {
        "status": "ok",
        "input": {
            "test_id": test_id,
            "student_id": student_id,
            "current_test_result_id": None,
        },
        "incorrect_questions": [],
        "total_questions_in_test": 0,
        "total_incorrect_questions": 0,
        "domain_performance": {
            "current": None,
            "history": None,
        },
        "notes": [],
    }

    # Check upstream
    if agent1_output.get("current_test_result") is None:
        result["status"] = "upstream_no_current_test"
        result["notes"].append(
            "Agent 1 did not return a current_test_result; skipping incorrect question extraction."
        )
        if agent1_output.get("notes"):
            result["notes"].extend(
                [f"[Agent1] {n}" for n in agent1_output["notes"]]
            )
        return result

    current = agent1_output["current_test_result"]
    current_test_result_id = current["id"]
    result["input"]["current_test_result_id"] = current_test_result_id

    # Load CSVs
    df_q = pd.read_csv(question_path)
    df_a = pd.read_csv(answer_path)
    df_tq = pd.read_csv(tq_path)
    df_ta = pd.read_csv(ta_path)

    # Filter to this test_result
    df_tq_current = df_tq[df_tq["examResultId"] == current_test_result_id].copy()
    if df_tq_current.empty:
        result["status"] = "no_question_results_for_test"
        result["notes"].append(
            f"No test_question_result rows found for TestResultId={current_test_result_id}."
        )
        return result

    result["total_questions_in_test"] = int(len(df_tq_current))

    df_ta_current = df_ta[
        df_ta["examResultQuestionId"].isin(df_tq_current["id"])
    ].copy()
    if df_ta_current.empty:
        result["status"] = "no_answers_for_test"
        result["notes"].append(
            "No test_answer_result rows found for the current test_result (no answers logged)."
        )
        return result

    # Domain performance for current attempt
    result["domain_performance"]["current"] = _build_domain_performance(
        df_tq_subset=df_tq_current,
        df_ta=df_ta,
        df_q=df_q,
    )

    # Domain performance for previous attempt (if available)
    history = agent1_output.get("history_test_result")
    if history and history.get("id"):
        history_id = history["id"]
        df_tq_history = df_tq[df_tq["examResultId"] == history_id].copy()
        if not df_tq_history.empty:
            result["domain_performance"]["history"] = _build_domain_performance(
                df_tq_subset=df_tq_history,
                df_ta=df_ta,
                df_q=df_q,
            )

    #Aggregate per-question to catch any incorrect attempts & keep all answers of that question
    agg = (
        df_ta_current.groupby("examResultQuestionId")
        .agg(
            any_incorrect=("isCorrect", lambda s: (~s.astype(bool)).any()),
            student_answers=("answerValue", lambda s: list(s)),
        )
        .reset_index()
    )

    df_join = df_tq_current.merge(
        agg,
        left_on="id",
        right_on="examResultQuestionId",
        how="left",
    )

    df_incorrect = df_join[df_join["any_incorrect"] == True].copy()
    if df_incorrect.empty:
        result["status"] = "no_incorrect_answers"
        result["notes"].append(
            "All questions in this test were answered correctly; no incorrect questions to analyze."
        )
        return result

    result["total_incorrect_questions"] = int(len(df_incorrect))

    # Attach question bank info
    df_incorrect = df_incorrect.merge(
        df_q,
        left_on="questionId",
        right_on="id",
        how="left",
        suffixes=("_tq", "_q"),
    )

    # Build answer lookups
    df_correct = df_a[df_a["isCorrect"] == True].copy()
    correct_lookup = (
        df_correct.groupby("questionId")["value"].apply(list).to_dict()
    )
    all_answers_lookup = (
        df_a.groupby("questionId")["value"].apply(list).to_dict()
    )

    incorrect_questions: List[Dict[str, Any]] = []
    tq_id_col = "id_tq" if "id_tq" in df_incorrect.columns else "id"

    for _, row in df_incorrect.iterrows():
        qid = row["questionId"]
        test_result_question_id = row[tq_id_col]
        incorrect_questions.append(
            {
                "questionId": qid if pd.notna(qid) else None,
                "testResultQuestionId": test_result_question_id,
                "questionText": row.get("question", None),
                "explanation": row.get("explanation", None),
                "studentAnswers": row["student_answers"],
                "correctAnswers": correct_lookup.get(qid, []),
                "allAnswers": all_answers_lookup.get(qid, []),
                "difficulty": row.get("difficulty", None),
                "score": row.get("score", None),
            }
        )

    result["incorrect_questions"] = incorrect_questions

    if not incorrect_questions:
        result["status"] = "incorrect_questions_not_built"
        result["notes"].append(
            "Unexpected: df_incorrect had rows but no incorrect_questions were built."
        )

    return result


def _build_domain_performance(
    df_tq_subset: pd.DataFrame,
    df_ta: pd.DataFrame,
    df_q: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute per-domain accuracy for a given exam attempt.
    """
    if df_tq_subset.empty:
        return None

    df_ta_subset = df_ta[
        df_ta["examResultQuestionId"].isin(df_tq_subset["id"])
    ].copy()
    if df_ta_subset.empty:
        return None

    agg = (
        df_ta_subset.groupby("examResultQuestionId")
        .agg(any_incorrect=("isCorrect", lambda s: (~s.astype(bool)).any()))
        .reset_index()
    )

    df_join = df_tq_subset.merge(
        agg,
        left_on="id",
        right_on="examResultQuestionId",
        how="left",
    ).merge(
        df_q[["id", "domain"]],
        left_on="questionId",
        right_on="id",
        how="left",
        suffixes=("", "_q"),
    )

    df_join["is_correct"] = ~(df_join["any_incorrect"].fillna(True))
    df_join["domain"] = df_join["domain"].fillna("Unknown")

    domain_rows = []
    total_correct = 0
    total_questions = 0
    for domain, grp in df_join.groupby("domain"):
        total = len(grp)
        correct = int(grp["is_correct"].sum())
        incorrect = total - correct
        accuracy = correct / total if total else None
        domain_rows.append(
            {
                "domain": domain,
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": accuracy,
            }
        )
        total_questions += total
        total_correct += correct

    overall_accuracy = total_correct / total_questions if total_questions else None
    return {
        "domains": domain_rows,
        "overall": {
            "total": total_questions,
            "correct": total_correct,
            "incorrect": total_questions - total_correct,
            "accuracy": overall_accuracy,
        },
    }
