# pipeline/run_pipeline.py
import os
import time
from functools import wraps
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# --- Ensure project root is on sys.path (works both locally & in Docker) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.agent1_test_context import get_student_test_history
from agents.agent2_incorrect_questions import get_incorrect_question_cases
from agents.agent3_weakness_extraction import extract_weaknesses_and_patterns
from agents.agent4_course_recommendation import recommend_courses_for_student
from agents.agent5_user_facing_response import generate_user_facing_response

from config import (
    QUESTION_PATH,
    ANSWER_PATH,
    TQ_PATH,
    TA_PATH,
    TEST_RESULT_PATH,
    TEST_ID, 
    STUDENT_ID,
    MAX_COURSES,
    PARTICIPANT_RANKING,
    RUN_LOG_PATH,
    Course,
    CourseScore,
    Weakness,
)
from pipeline.run_logging import reset_token_log, get_token_entries

def log_call(func):
    """Decorator that reports runtime for each function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f"[Runtime] Calling {func.__name__}")
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[Runtime] {func.__name__} finished in {elapsed:.2f}s")
    return wrapper


# --------------------------------------------------------------------
# Full Pipeline
# --------------------------------------------------------------------
@log_call
def run_full_pipeline(
    test_id: str,
    student_id: str,
    max_courses: int = 5,
    participant_ranking: float | None = None,
    language: str = "EN",
    rerank_courses: bool = True,
) -> Dict[str, Any]:
    reset_token_log()

    # ---------------- Agent 1 ----------------
    t_agent1 = time.perf_counter()
    agent1_out = get_student_test_history(
        test_id=test_id,
        student_id=student_id,
        csv_path=TEST_RESULT_PATH,
    )
    # no test taken
    if agent1_out.get("current_test_result") is None:
        return {
            "status": "agent1_error",
            "agent1_output": agent1_out,
            "message": "No current test for this student/test_id.",
        }
    print(f"Agent 1 completed successfully in {time.perf_counter() - t_agent1:.2f}s")

    # ---------------- Agent 2 ----------------
    t_agent2 = time.perf_counter()
    agent2_out = get_incorrect_question_cases(
        agent1_out,
        question_path=QUESTION_PATH,
        answer_path=ANSWER_PATH,
        tq_path=TQ_PATH,
        ta_path=TA_PATH,
    )

    if agent2_out["status"] != "ok" and agent2_out["status"] != "no_incorrect_answers":
        return {
            "status": "agent2_error",
            "agent1_output": agent1_out,
            "agent2_output": agent2_out,
        }

    incorrect_cases = agent2_out["incorrect_questions"]
    all_correct = not incorrect_cases
    print(f"Agent 2 completed successfully in {time.perf_counter() - t_agent2:.2f}s")

    weaknesses_llm = []
    course_rec_output = {"weaknesses": [], "recommendations": []}

    if not all_correct:
        # ---------------- Agent 3 ----------------
        t_agent3 = time.perf_counter()
        weaknesses_llm = extract_weaknesses_and_patterns(incorrect_cases)
        if not weaknesses_llm:
            return {
                "status": "no_weaknesses",
                "agent1_output": agent1_out,
                "agent2_output": agent2_out,
                "weaknesses_raw": [],
            }  
        print(f"Agent 3:WeaknessExtraction completed successfully in {time.perf_counter() - t_agent3:.2f}s")

        # ---------------- Agent 4 ----------------
        t_agent4 = time.perf_counter()
        course_rec_output = None
        try:
            print("Agent 4 – vector search recommendation...")
            course_rec_output = recommend_courses_for_student(
                weaknesses_raw=weaknesses_llm,
                max_courses_pr_weakness=max_courses,
                rerank_enabled=rerank_courses,
            )
            print(f"Agent 4 completed vector search successfully in {time.perf_counter() - t_agent4:.2f}s")
        except Exception as e:
            print(e)
            print(f"[WARN] Vector search failed: {e}")

    # ---------------- Agent 5 ----------------
    t_agent5 = time.perf_counter()
    if not all_correct and (not course_rec_output or not course_rec_output.get("recommendations")):
        print("[WARN] No course recommendations available for LLM summary.")
        return {
            "status": "no_course_recommendations",
            "agent1_output": agent1_out,
            "agent2_output": agent2_out,
            "weaknesses_llm": weaknesses_llm,
            "course_recommendation": course_rec_output,
        }
    weaknesses = course_rec_output["weaknesses"]
    recommendations = course_rec_output["recommendations"]

    result = generate_user_facing_response(
        weaknesses=weaknesses,
        recommendations=recommendations,
        test_result=agent1_out.get("current_test_result"),
        history_result=agent1_out.get("history_test_result"),
        incorrect_summary={
            "total_questions_in_test": agent2_out.get("total_questions_in_test"),
            "total_incorrect_questions": agent2_out.get("total_incorrect_questions"),
            "notes": agent2_out.get("notes"),
            "status": agent2_out.get("status"),
        },
        all_correct=all_correct,
        participant_ranking=participant_ranking,
        domain_performance=agent2_out.get("domain_performance"),
        language=language,
    )

    print("Response in : ", language)
    print(f"Agent 5 completed successfully in {time.perf_counter() - t_agent5:.2f}s")

    status_val = "ok" if not all_correct else "ok_all_correct"
    _write_run_log(
        status=status_val,
        agent1_output=agent1_out,
        agent2_output=agent2_out,
        weaknesses_llm=weaknesses_llm,
        course_recommendation=course_rec_output,
        participant_ranking=participant_ranking,
        language=language,
        rerank_courses=rerank_courses,
        final_response=result,
    )

    return {
        "user_facing_paragraph": result.get("user_facing_paragraph", ""),
    }


# --------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------
def _write_run_log(**payload: Any) -> None:
    """
    Append a JSON entry with run metadata, agent outputs, final response, and token log snapshot.
    """
    log_file = Path(RUN_LOG_PATH)
    run_entry_raw = {
        "run_datetime": datetime.now(timezone.utc).isoformat(),
        **payload,
        "token_log": get_token_entries(),
    }
    run_entry = _simplify_for_json(run_entry_raw)
    entries = _read_run_log_entries(log_file)
    entries.append(run_entry)
    log_file.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_run_log_entries(log_file: Path) -> list[Any]:
    if not log_file.exists():
        return []
    try:
        return json.loads(log_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # fallback: try JSONL-style each line
        lines = [ln.strip() for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        entries = []
        for ln in lines:
            try:
                entries.append(json.loads(ln))
            except Exception:
                continue
        return entries


def _simplify_for_json(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable structures.
    - Weakness -> dict with key fields
    - Course/CourseScore -> dict
    - Fallback -> string representation
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Weakness):
        return {
            "id": obj.id,
            "text": obj.text,
            "importance": obj.importance,
            "metadata": obj.metadata,
        }
    if isinstance(obj, Course):
        return {
            "id": obj.id,
            "lesson_title": obj.lesson_title,
            "description": obj.description,
            "link": obj.link,
            "metadata": obj.metadata,
        }
    if isinstance(obj, CourseScore):
        return {
            "course": _simplify_for_json(obj.course),
            "weakness_id": obj.weakness_id,
            "score": obj.score,
            "reason": obj.reason,
        }
    if isinstance(obj, dict):
        return {k: _simplify_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_simplify_for_json(v) for v in obj]
    return str(obj)


# --------------------------------------------------------------------
# Run manually
# --------------------------------------------------------------------
if __name__ == "__main__":

    result = run_full_pipeline(
        test_id=TEST_ID,
        student_id=STUDENT_ID,
        max_courses=MAX_COURSES,
        participant_ranking=PARTICIPANT_RANKING,
        language="EN",
        rerank_courses=True,
    )

    print(result)
