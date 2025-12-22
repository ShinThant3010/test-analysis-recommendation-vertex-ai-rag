# agents/agent5_user_facing_response.py
from typing import Any, Dict, List, Optional
import json
import time
from config import (
    GENERATION_MODEL,
    client,
    Course,
    Weakness,
    CourseScore,
    PARTICIPANT_RANKING,

)
from token_logger import log_token_usage, extract_token_counts

def generate_user_facing_response(
    weaknesses: List[Weakness],
    recommendations: List[CourseScore],
    test_result: Optional[Dict[str, Any]] = None,
    history_result: Optional[Dict[str, Any]] = None,
    incorrect_summary: Optional[Dict[str, Any]] = None,
    all_correct: bool = False,
    participant_ranking: Optional[float] = PARTICIPANT_RANKING,
    domain_performance: Optional[Dict[str, Any]] = None,
    language: str = "EN",
) -> Dict[str, Any]:
    """
    Generate a narrative performance report. The model is allowed to infer the domain
    ONLY if domain clues appear in the weakness descriptions, test name, or course titles.
    Otherwise, it must stay domain-neutral.
    """

    # Fast path: no incorrect answers, so skip LLM.
    if all_correct:
        summary_json = _congrats_summary(
            test_result=test_result,
            history_result=history_result,
            ranking_sentence=_ranking_sentence(participant_ranking),
            progress_heading=_progress_heading(test_result, history_result),
        )
        return {
            "summary": summary_json,
            "recommendations": [],
        }

    weaknesses_text = "\n".join(
        f"- ({w.id}) {w.text} (importance={w.importance})"
        for w in weaknesses
    )

    recs_text = "\n".join(
        f"- {cs.course.lesson_title} (id={cs.course.id}) helps weakness {cs.weakness_id}"
        for cs in recommendations
    )

    test_result_text = json.dumps(test_result or {}, ensure_ascii=False, indent=2)
    history_result_text = json.dumps(history_result or {}, ensure_ascii=False, indent=2)
    incorrect_summary_text = json.dumps(incorrect_summary or {}, ensure_ascii=False, indent=2)
    ranking_text = _ranking_value_for_prompt(participant_ranking)
    domain_perf_text = json.dumps(domain_performance or {}, ensure_ascii=False, indent=2)
    progress_heading = _progress_heading(test_result, history_result) or "N/A"
    test_title = _test_title(test_result, history_result) or "N/A"
    language_text = language or "EN"

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

        Full test result for the CURRENT attempt (use this for "Current Performance"):
        {test_result_text}

        Previous attempt (if any):
        {history_result_text}

        Incorrect-question summary:
        {incorrect_summary_text}

        Participant ranking (optional, only if provided and cohort >= 100). The value is fractional (e.g., 0.317 means top 31.7%):
        {ranking_text}

        Domain performance by attempt (if history is present, compare current vs previous):
        {domain_perf_text}

        Heading to use before the domain comparison if history exists:
        {progress_heading}

        Weaknesses identified:
        {weaknesses_text}

        Selected recommended courses (do NOT change this list):
        {recs_text}

        --- REQUIRED OUTPUT FORMAT (JSON ONLY) ---
        {{
            "Test Title": "<the current test title>",
            "Current Performance": "<one short paragraph summarizing current ability. Mention domain only if clearly detectable. Summarize strengths and weaknesses clearly.>",
            "Area to be Improved": "<one short paragraph describing the key skills or behaviors that need focus.>",
            "Recommended Course": [
                "<Course A explanation referencing one of the provided courses>",
                "<Course B explanation referencing one of the provided courses>",
                "..."
            ],
            "Progress Compared to Previous Test": "<Use the provided heading if history exists, otherwise empty string.>",
            "Domain Comparison": [
                "<Domain A: Improved by +X% (short reasoning)>",
                "<Domain B: Declined by -Y% (short reasoning)>"
            ]
        }}
        --- TONE & FORMAT ---

        - Use a supportive and encouraging tone.
        - Keep each section concise (2-4 sentences).
        - Focus on clarity and actionable insights.
        - Write in smooth, narrative paragraphs.
        - Base "Current Performance" on the provided test result fields (score, attempts, status, totals) plus the incorrect-question summary; do NOT rely solely on weaknesses.
        - If participant ranking is provided (not N/A), include a short ranking statement using that value.
        - If domain performance includes both current and history, add a concise domain-wise comparison highlighting improvements or declines; otherwise omit or leave the array empty.
        - Include the test title at the start via the \"Test Title\" field.
        - If there is a previous attempt, set \"Progress Compared to Previous Test\" to the heading provided above; otherwise set it to an empty string.
        - Respond in the requested language: {language_text} (supported: EN, TH). Keep JSON keys in English.
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
        print("[WARN] LLM response invalid or missing JSON, falling back to default summary.")
        summary_json = _fallback_summary(
            weaknesses=weaknesses,
            recommendations=recommendations,
            test_result=test_result,
            incorrect_summary=incorrect_summary,
            history_result=history_result,
            ranking_sentence=_ranking_sentence(participant_ranking),
            domain_performance=domain_performance,
            progress_heading=_progress_heading(test_result, history_result),
            test_title=test_title,
        )
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
    test_result: Optional[Dict[str, Any]] = None,
    incorrect_summary: Optional[Dict[str, Any]] = None,
    history_result: Optional[Dict[str, Any]] = None,
    ranking_sentence: str = "",
    domain_performance: Optional[Dict[str, Any]] = None,
    progress_heading: str = "",
    test_title: str = "",
) -> Dict[str, Any]:
    current_perf_parts = []
    if test_result:
        current_perf_parts.append(_summarize_test_result(test_result, incorrect_summary))
    if history_result:
        history_sentence = _summarize_history(history_result)
        if history_sentence:
            current_perf_parts.append(history_sentence)
    if ranking_sentence:
        current_perf_parts.append(ranking_sentence)

    current_perf = (
        " ".join(p for p in current_perf_parts if p) or
        "We reviewed your recent performance and identified specific skills that would "
        "benefit from additional focus."
    )

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

    domain_comparison = _domain_improvement_summaries(domain_performance)

    return {
        "Test Title": test_title or (test_result.get("testTitle") if test_result else ""),
        "Current Performance": current_perf,
        "Area to be Improved": (
            f"Priority focus areas include {weakness_titles}. Strengthening these abilities "
            "will improve overall consistency."
        ),
        "Recommended Course": rec_sentences,
        "Progress Compared to Previous Test": progress_heading if domain_comparison else "",
        "Domain Comparison": domain_comparison,
    }


def _congrats_summary(
    test_result: Optional[Dict[str, Any]],
    history_result: Optional[Dict[str, Any]],
    ranking_sentence: str = "",
    progress_heading: str = "",
) -> Dict[str, Any]:
    perf_sentence = ""
    if test_result:
        perf_sentence = _summarize_test_result(test_result, incorrect_summary={"total_questions_in_test": None, "total_incorrect_questions": 0})
    history_sentence = _summarize_history(history_result) if history_result else ""

    current_perf = " ".join(
        part for part in [perf_sentence, history_sentence, ranking_sentence, "Congratulations on answering every question correctly!"]
        if part
    )
    if not current_perf:
        current_perf = "Congratulations on answering every question correctly!"

    return {
        "Test Title": _test_title(test_result, history_result),
        "Current Performance": current_perf,
        "Area to be Improved": "You achieved full accuracy on this attempt. Continue practicing to maintain your performance.",
        "Recommended Course": [],
        "Progress Compared to Previous Test": "" if not progress_heading else progress_heading,
        "Domain Comparison": [],
    }


def _summarize_test_result(
    test_result: Dict[str, Any],
    incorrect_summary: Optional[Dict[str, Any]] = None,
) -> str:
    title = test_result.get("testTitle") or "this test"
    attempt = _to_int(test_result.get("attemptNumber"))
    total_attempts = _to_int(test_result.get("totalAttempts"))
    attempt_clause = ""
    if attempt:
        if total_attempts and total_attempts >= attempt:
            attempt_clause = f" (attempt {attempt} of {total_attempts})"
        else:
            attempt_clause = f" (attempt {attempt})"

    score = _to_int(test_result.get("earnedScore"))
    total_score = _to_int(test_result.get("totalScore"))
    status = test_result.get("status")

    score_clause = None
    if score is not None and total_score:
        percent = (score / total_score) * 100
        score_clause = f"scored {score}/{total_score} ({percent:.0f}%)"
    elif score is not None:
        score_clause = f"scored {score}"

    parts: List[str] = [f"In {title}{attempt_clause}"]
    if score_clause:
        parts.append(score_clause)
    if status:
        parts.append(f"status: {status}")

    if incorrect_summary:
        total_q = _to_int(incorrect_summary.get("total_questions_in_test"))
        incorrect_q = _to_int(incorrect_summary.get("total_incorrect_questions"))
        if total_q:
            if incorrect_q is None:
                parts.append(f"{total_q} questions assessed.")
            else:
                accuracy = None
                if incorrect_q is not None:
                    accuracy = (max(total_q - incorrect_q, 0) / total_q) * 100
                parts.append(
                    f"{total_q} questions with {incorrect_q} incorrect"
                    + (f" ({accuracy:.0f}% correct)" if accuracy is not None else "")
                )

    return "; ".join(parts) + "."


def _summarize_history(history_result: Dict[str, Any]) -> str:
    attempt = _to_int(history_result.get("attemptNumber"))
    score = _to_int(history_result.get("earnedScore"))
    total_score = _to_int(history_result.get("totalScore"))

    if score is None or total_score is None:
        if attempt:
            return f"Previous attempt {attempt} is on record."
        return ""

    percent = (score / total_score) * 100 if total_score else None
    attempt_text = f"attempt {attempt} " if attempt else ""
    percent_text = f" ({percent:.0f}%)" if percent is not None else ""
    return f"Previous {attempt_text}scored {score}/{total_score}{percent_text}."


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ranking_sentence(participant_ranking: Optional[float]) -> str:
    if participant_ranking is None:
        return ""
    try:
        pct = participant_ranking * 100 if participant_ranking <= 1 else participant_ranking
        return f"Ranked within the top {pct:.1f}% of participants."
    except Exception:
        return ""


def _ranking_value_for_prompt(participant_ranking: Optional[float]) -> str:
    if participant_ranking is None:
        return "N/A"
    try:
        pct = participant_ranking * 100 if participant_ranking <= 1 else participant_ranking
        return f"{participant_ranking} (approx. top {pct:.1f}% of participants)"
    except Exception:
        return "N/A"


def _domain_improvement_summaries(
    domain_performance: Optional[Dict[str, Any]]
) -> List[str]:
    if not domain_performance:
        return []

    current = (domain_performance or {}).get("current") or {}
    history = (domain_performance or {}).get("history") or {}

    curr_domains = {d["domain"]: d for d in (current.get("domains") or []) if "domain" in d}
    hist_domains = {d["domain"]: d for d in (history.get("domains") or []) if "domain" in d}

    summaries: List[str] = []
    for domain, curr in curr_domains.items():
        if domain not in hist_domains:
            continue
        hist = hist_domains[domain]
        curr_acc = curr.get("accuracy")
        hist_acc = hist.get("accuracy")
        if curr_acc is None or hist_acc is None:
            continue
        delta = (curr_acc - hist_acc) * 100
        direction = "Improved" if delta >= 0 else "Declined"
        summaries.append(
            f"{domain}: {direction} by {delta:+.1f}% (from {hist_acc*100:.1f}% to {curr_acc*100:.1f}%)"
        )
    return summaries


def _progress_heading(
    test_result: Optional[Dict[str, Any]],
    history_result: Optional[Dict[str, Any]],
) -> str:
    if not history_result:
        return ""
    title = _test_title(test_result, history_result)
    if not title:
        title = "previous test"
    return f"Progress Compared to Previous Test ({title}):"


def _test_title(
    test_result: Optional[Dict[str, Any]],
    history_result: Optional[Dict[str, Any]] = None,
) -> str:
    return (
        (test_result or {}).get("testTitle")
        or (history_result or {}).get("testTitle")
        or ""
    )
