# agents/agent5_user_facing_response.py
from typing import Any, Dict, List
import json
import time
from config import (
    GENERATION_MODEL,
    client,
    Course,
    Weakness,
    CourseScore,

)
from token_logger import log_token_usage, extract_token_counts

def generate_user_facing_response(
    weaknesses: List[Weakness],
    recommendations: List[CourseScore],
) -> Dict[str, Any]:
    """
    Generate a narrative performance report. The model is allowed to infer the domain
    ONLY if domain clues appear in the weakness descriptions, test name, or course titles.
    Otherwise, it must stay domain-neutral.
    """

    weaknesses_text = "\n".join(
        f"- ({w.id}) {w.text} (importance={w.importance})"
        for w in weaknesses
    )

    recs_text = "\n".join(
        f"- {cs.course.lesson_title} (id={cs.course.id}) helps weakness {cs.weakness_id}"
        for cs in recommendations
    )

    # === JSON Prompt === #
    prompt = f"""
        You are generating a concise JSON report for a student or professional
        based on their weaknesses and recommended courses.

        DOMAIN INFERENCE RULE:
        - You MAY infer the domain **only if** the weaknesses, course titles, or metadata
          clearly indicate a domain (e.g., English listening, SQL queries, financial modeling,
          safety engineering, supply chain forecasting).
        - If domain is NOT clearly indicated, stay domain-neutral and avoid assuming a specific field.

        DO NOT create fictional exams, fake metrics, or imaginary organizations.
        Use only information from the weaknesses and course list.

        --- INPUT DATA ---

        Weaknesses identified:
        {weaknesses_text}

        Selected recommended courses (do NOT change this list):
        {recs_text}

        --- REQUIRED OUTPUT FORMAT (JSON ONLY) ---
        {{
            "Current Performance": "<one short paragraph summarizing current ability. Mention domain only if clearly detectable. Summarize strengths and weaknesses clearly.>",
            "Area to be Improved": "<one short paragraph describing the key skills or behaviors that need focus.>",
            "Recommended Course": [
                "<Course A explanation referencing one of the provided courses>",
                "<Course B explanation referencing one of the provided courses>",
                "..."
            ]
        }}
        --- TONE & FORMAT ---

        - Use a supportive and encouraging tone.
        - Keep each section concise (2-4 sentences).
        - Focus on clarity and actionable insights.
        - Write in smooth, narrative paragraphs.
        - Return ONLY valid JSON (no code fences, no commentary).
        - The "Recommended Course" array must describe each provided course and how it supports the weaknesses.
        - Do not invent new courses or change their titles.
        """
    
    
    # === Call Gemini === #
    response = None
    start = time.time()
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
        )
        raw_text = (response.text or "").strip()
        if not raw_text:
            raise ValueError("Empty response")
        summary_json = _parse_llm_json(raw_text)
        if not summary_json:
            raise ValueError("Unable to parse JSON")

    except Exception:
        summary_json = _fallback_summary(weaknesses, recommendations)
    finally:
        elapsed = time.time() - start
        input_tokens, output_tokens = extract_token_counts(response) if response else (None, None)
        log_token_usage(
            usage="agent5: user-facing response generation",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            runtime_seconds=elapsed,
        )

    # === Build simple recommendations JSON === #
    rec_list = [
        {
            "course_id": cs.course.id,
            "course_title": cs.course.lesson_title,
            "target_weakness_id": cs.weakness_id,
            "explanation": cs.reason or "",
        }
        for cs in recommendations
    ]

    return {
        "summary": summary_json,
        "recommendations": rec_list,
    }


def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def _fallback_summary(
    weaknesses: List[Weakness],
    recommendations: List[CourseScore],
) -> Dict[str, Any]:
    if weaknesses:
        weakness_titles = ", ".join(w.text for w in weaknesses[:3])
    else:
        weakness_titles = "the assessed skills"

    rec_sentences = [
        f"{cs.course.lesson_title} targets weakness {cs.weakness_id}."
        for cs in recommendations
    ]
    if not rec_sentences:
        rec_sentences = ["No course recommendations were generated."]

    return {
        "Current Performance": (
            "We reviewed your recent performance and identified specific skills that would "
            "benefit from additional focus."
        ),
        "Area to be Improved": (
            f"Priority focus areas include {weakness_titles}. Strengthening these abilities "
            "will improve overall consistency."
        ),
        "Recommended Course": rec_sentences,
    }
