# agents/agent4_course_recommendation.py

from __future__ import annotations

import csv
import uuid
from typing import Any, Dict, List

from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
import vertexai
from google import genai
from google.genai.types import EmbedContentConfig


from config import (
    COURSE_CSV_PATH,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEPLOYED_INDEX_ID,
    INDEX_ENDPOINT_NAME,
    ENDPOINT_DISPLAY_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    Course,
    Weakness,
    CourseScore,
)

# Initialize Vertex AI and GenAI client
vertexai.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
genai_client = genai.Client()

def embed_texts(texts: List[str], dim: int = EMBEDDING_DIMENSION) -> List[List[float]]:
    """Embed texts in batches to respect 100-request limit."""
    batch_size = 100
    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = genai_client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=batch,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=dim
            ),
        )
        all_vectors.extend([e.values for e in resp.embeddings])
    return all_vectors

def _load_course_lookup() -> Dict[str, Dict[str, Any]]:
    if not COURSE_CSV_PATH.exists():
        raise FileNotFoundError(f"Course CSV not found at {COURSE_CSV_PATH}")

    course_lookup: Dict[str, Dict[str, Any]] = {}
    with COURSE_CSV_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            course_id = row.get("id")
            if not course_id:
                continue
            course_lookup[course_id] = row
    return course_lookup

def _parse_weaknesses(weaknesses_raw: List[Dict[str, Any]]) -> List[Weakness]:
    weaknesses: List[Weakness] = []
    for w in weaknesses_raw:
        w_id = w.get("id") or str(uuid.uuid4()) # no id, generate one
        text = w["weakness"]
        importance = float(w.get("importance", 1.0))
        meta = {k: v for k, v in w.items() if k not in ["id", "text", "importance"]} # will be pattern, description, questions, freq
        weaknesses.append(Weakness(id=w_id, text=text, importance=importance, metadata=meta))
    return weaknesses

def _query_vertex_index(query_text: str, limit: int) -> List[Any]:
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
    endpoint_name = ""
    for ep in endpoints:
        if ep.display_name == ENDPOINT_DISPLAY_NAME:
            endpoint_name = ep.resource_name

    if not endpoint_name:
        raise ValueError(f"Matching Engine endpoint with display name '{ENDPOINT_DISPLAY_NAME}' not found.")    
        return []
    
    endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
    query_vector = embed_texts([query_text])[0]

    print(query_vector[:10])  # Print first 10 dimensions for debugging
    neighbors = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=limit,
        return_full_datapoint=False,
    )
    return neighbors[0] if neighbors else []

def recommend_courses_for_student(
    weaknesses_raw: List[Dict[str, Any]],
    max_courses_pr_weakness: int = 5,
) -> Dict[str, Any]:
    """
    Fast online path:
    - assumes Vertex Matching Engine index is already deployed
    - embeds each weakness and queries deployed endpoint for nearest courses
    """

    weaknesses = _parse_weaknesses(weaknesses_raw)
    course_lookup = _load_course_lookup()

    all_recommendations: List[CourseScore] = []

    for w in weaknesses:
        print(f"Querying courses for weakness: {w.id} - {w.text[:60]}...")
        neighbors = _query_vertex_index(w.text, max_courses_pr_weakness)

        for neighbor in neighbors:
            course_id = str(neighbor.id)
            metadata = course_lookup.get(course_id, {})
            lesson_title = metadata.get("lesson_title") or "Untitled course"
            desc = metadata.get("description") or ""
            link = metadata.get("link") or metadata.get("course_url") or ""
            course = Course(
                id=course_id,
                lesson_title=lesson_title,
                description=desc,
                link=link,
                metadata=metadata,
            )
            distance = float(getattr(neighbor, "distance", 0.0) or 0.0)
            score = 1 / (1 + distance)
            reason = f"Retrieved by semantic match to weakness '{w.text[:80]}...'."

            all_recommendations.append(
                CourseScore(
                    course=course,
                    weakness_id=w.id,
                    score=score,
                    reason=reason,
                )
            )

    selected_recommendations = _select_final_courses(
        all_recommendations, max_total=5
    )

    response = {
        "weaknesses": weaknesses,
        "recommendations": selected_recommendations,
    }
    print("Course recommendation process completed.")
    return response


def _select_final_courses(
    all_recommendations: List[CourseScore],
    max_total: int,
) -> List[CourseScore]:
    if not all_recommendations:
        return []

    # Guarantee at most one top pick per weakness
    top_per_weakness: Dict[str, CourseScore] = {}
    for rec in all_recommendations:
        best = top_per_weakness.get(rec.weakness_id)
        if best is None or rec.score > best.score:
            top_per_weakness[rec.weakness_id] = rec

    selected = _dedupe_by_course(list(top_per_weakness.values()))
    selected.sort(key=lambda cs: cs.score, reverse=True)

    if len(selected) >= max_total:
        return selected[:max_total]

    remaining_slots = max_total - len(selected)
    remaining = [
        rec
        for rec in all_recommendations
        if rec not in selected
    ]
    remaining = _dedupe_by_course(remaining)
    remaining.sort(key=lambda cs: cs.score, reverse=True)

    return selected + remaining[:remaining_slots]


def _dedupe_by_course(recs: List[CourseScore]) -> List[CourseScore]:
    seen: set[str] = set()
    unique: List[CourseScore] = []
    for rec in recs:
        cid = rec.course.id
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(rec)
    return unique


    # response = {
    #     # plain-dict view for every object in weaknesses
    #     "weaknesses": [w.__dict__ for w in weaknesses],
    #     "recommendations": [
    #         {
    #             "course_id": cs.course.id,
    #             "course_title": cs.course.lesson_title,
    #             "weakness_id": cs.weakness_id,
    #             "score": cs.score,
    #             "reason": cs.reason,
    #             "course_metadata": cs.course.metadata,
    #         }
    #         for cs in all_recommendations_sorted
    #     ],
    # }
    # print("Course recommendation process completed.")
    # return response
