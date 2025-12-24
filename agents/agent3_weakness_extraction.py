# agents/agent3_weakness_extraction.py
from typing import List, Dict, Any
import json
import ast
import re
import ulid
import time

from config import client, GENERATION_MODEL
from pipeline.run_logging import log_token_usage, extract_token_counts


def extract_weaknesses_and_patterns(
    incorrect_cases: List[Dict[str, Any]],
    model_name: str = GENERATION_MODEL,
) -> List[Dict[str, Any]]:
    """
    Use Gemini to turn incorrect question cases into concrete weaknesses & patterns.
    """
    if not incorrect_cases:
        return []

    cases_json = json.dumps(incorrect_cases, ensure_ascii=False, indent=2)

    prompt = f"""
        You are a diagnostic engine for assessment tests across many domains
        (e.g., language exams, aptitude tests, professional certifications,
        and Thai civil service exams such as ข้อสอบ กพ).

        You receive a JSON array of questions where the student answered incorrectly.

        Task:
        1. Look across ALL incorrect questions for this single student and test.
        2. Find concrete, reusable weaknesses and error patterns (not just "Grammar" or "Math").
        3. Group evidence questions that share the same weakness or pattern.

        Output format (JSON ONLY, no extra text):

        [
        {{
            "weakness": "short name (1 sentence max, specific to the pattern)",
            "pattern_type": "language | numeracy | logical_reasoning | reading_comprehension | domain_knowledge | test_strategy | other",
            "description": "2–4 sentences explaining the pattern and why errors happen.",
            "evidence_question_ids": [<questionId>, ...],
            "frequency": <number of questions that show this pattern>."
        }}
        ]

        Here is the JSON array of incorrect questions:

        {cases_json}

        Respond with ONLY the JSON array as described above.
        """
    
    response = None
    start = time.time()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[{"parts": [{"text": prompt}]}],
        )
    finally:
        elapsed = time.time() - start
        input_tokens, output_tokens = extract_token_counts(response) if response else (None, None)
        log_token_usage(
            usage="agent3: weakness extraction",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            runtime_seconds=elapsed,
        )
    weaknesses = convert_llm_weaknesses_for_agent3(remove_code_fences(response.text))

    # Add ULID as id
    for d in weaknesses:
        d["id"] = str(ulid.new())
    return weaknesses

def remove_code_fences(text: str) -> str:
    text = text.replace("```json", "")
    text = text.replace("```", "")
    return text.strip()

def convert_llm_weaknesses_for_agent3(
    raw_text: str
) -> List[Dict[str, Any]]:
    """
    Parse LLM output into a list of weakness dicts.

    1) Try strict JSON.
    2) Try Python literal (e.g. [{'weakness': ...}]).
    3) Fallback: regex extraction using keywords.
    """

    text = raw_text.strip()

    # ---- 1) Try JSON ----
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        print('Failed to parse as JSON  - trying next method...')
        pass

    # ---- 2) Try Python literal (LLM returns Python-style dict/list) ----
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        print('Failed to parse as Python literal  - trying regex fallback...')
        pass

    # ---- 3) Regex fallback: extract fields from messy text ----
    return _extract_weakness_by_regex(text)

def _extract_weakness_by_regex(text: str) -> List[Dict[str, Any]]:
    """
    Very simple heuristic parser:
    - looks for fields: weakness, pattern_type, description, evidence_question_ids, frequency
    - returns one-item list (but you can extend to multiple blocks later).
    """

    block = text  # if you have many, you can split by two newlines etc.
    result: Dict[str, Any] = {}

    # weakness
    m = re.search(r"weakness\s*[:=-]\s*['\"]?(.+?)['\"]?(?:\n|,|$)", block, re.IGNORECASE)
    if m:
        result["weakness"] = m.group(1).strip()

    # pattern_type
    m = re.search(r"pattern_type\s*[:=-]\s*['\"]?(.+?)['\"]?(?:\n|,|$)", block, re.IGNORECASE)
    if m:
        result["pattern_type"] = m.group(1).strip()

    # description (grab everything after 'description' until next keyword)
    m = re.search(r"description\s*[:=-]\s*['\"]?(.+)", block, re.IGNORECASE | re.DOTALL)
    if m:
        desc = m.group(1)
        # Cut off at next known field label
        desc = re.split(
            r"\b(evidence_question_ids|frequency|weakness|pattern_type)\b\s*[:=-]",
            desc,
        )[0]
        # Clean newlines + extra spaces
        desc = " ".join(desc.split())
        result["description"] = desc

    # evidence_question_ids: [1575, 123, ...]
    m = re.search(
        r"evidence_question_ids\s*[:=-]\s*\[([^\]]*)\]",
        block,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        nums = re.findall(r"\d+", m.group(1))
        if nums:
            result["evidence_question_ids"] = [int(n) for n in nums]

    # frequency
    m = re.search(r"frequency\s*[:=-]\s*(\d+)", block, re.IGNORECASE)
    if m:
        result["frequency"] = int(m.group(1))
    
    # Return as list (for consistent shape)
    return [result] if result else []
