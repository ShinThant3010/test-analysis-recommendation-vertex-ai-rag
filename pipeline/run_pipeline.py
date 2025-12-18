# pipeline/run_pipeline.py
import os
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
    COURSE_COLLECTION_NAME,
    PERSIST_DIRECTORY,
    TEST_ID, 
    STUDENT_ID,
    MAX_COURSES,
)

# --------------------------------------------------------------------
# Full Pipeline
# --------------------------------------------------------------------
def run_full_pipeline(
    test_id: int,
    student_id: int,
    max_courses: int = 5,
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COURSE_COLLECTION_NAME,
) -> Dict[str, Any]:

    # ---------------- Agent 1 ----------------
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
    print("Agent 1 completed successfully.")

    # ---------------- Agent 2 ----------------
    agent2_out = get_incorrect_question_cases(
        agent1_out,
        question_path=QUESTION_PATH,
        answer_path=ANSWER_PATH,
        tq_path=TQ_PATH,
        ta_path=TA_PATH,
    )

    if agent2_out["status"] != "ok":
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
    print("Agent 2 completed successfully.")

    # ---------------- Agent 3 ----------------
    weaknesses_llm = extract_weaknesses_and_patterns(incorrect_cases)
    if not weaknesses_llm:
        return {
            "status": "no_weaknesses",
            "agent1_output": agent1_out,
            "agent2_output": agent2_out,
            "weaknesses_raw": [],
        }
    print("Agent 3 - weakness extraction completed.")    
    print(weaknesses_llm)

    course_rec_output = None
    try:
        print("Agent 4 – vector search recommendation...")
        course_rec_output = recommend_courses_for_student(
            weaknesses_raw=weaknesses_llm,
            max_courses_pr_weakness=max_courses,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        print("Agent 4 completed vector search.")
    except Exception as e:
        print(f"[WARN] Vector search failed: {e}")

    # ---------------- Agent 5 ----------------
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
        persist_directory=PERSIST_DIRECTORY,
        collection_name="courses",
    )

    print(result)