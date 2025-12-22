# pipeline/run_pipeline.py
import os
import time
from functools import wraps
import sys
import json
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
)

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
) -> Dict[str, Any]:

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

    if agent2_out["status"] != "ok" or "no_incorrect_answers":
        return {
            "status": "agent2_error",
            "agent1_output": agent1_out,
            "agent2_output": agent2_out,
        }

    incorrect_cases = agent2_out["incorrect_questions"]
    if not incorrect_cases:
        return {
            "status": "no_incorrect_questions",
            "agent1_output": agent1_out,
            "agent2_output": agent2_out,
        }
    print(f"Agent 2 completed successfully in {time.perf_counter() - t_agent2:.2f}s")

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
        )
        print(f"Agent 4 completed vector search successfully in {time.perf_counter() - t_agent4:.2f}s")
    except Exception as e:
        print(e)
        print(f"[WARN] Vector search failed: {e}")

    # ---------------- Agent 5 ----------------
    t_agent5 = time.perf_counter()
    if not course_rec_output or not course_rec_output.get("recommendations"):
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
    )
    print(f"Agent 5 completed successfully in {time.perf_counter() - t_agent5:.2f}s")

    return {
        "status": "ok",
        "agent1_output": agent1_out,
        "agent2_output": agent2_out,
        "weaknesses_llm": weaknesses_llm,
        "course_recommendation": course_rec_output,
        "final_response": result,
    }


# --------------------------------------------------------------------
# Run manually
# --------------------------------------------------------------------
if __name__ == "__main__":

    result = run_full_pipeline(
        test_id=TEST_ID,
        student_id=STUDENT_ID,
        max_courses=MAX_COURSES,
    )

    print(result)