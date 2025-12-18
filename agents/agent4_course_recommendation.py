# agents/agent4_course_recommendation.py

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import uuid
import chromadb

from config import (
    PERSIST_DIRECTORY, 
    COURSE_COLLECTION_NAME,
    Course,
    Weakness,
    CourseScore,
)
from agents.gemini_embeddings import get_gemini_embedding_function

# ====== 2) ONLINE PATH: query only, no upsert ======

def _parse_weaknesses(weaknesses_raw: List[Dict[str, Any]]) -> List[Weakness]:
    weaknesses: List[Weakness] = []
    for w in weaknesses_raw:
        w_id = w.get("id") or str(uuid.uuid4()) # no id, generate one
        text = w["weakness"]
        importance = float(w.get("importance", 1.0))
        meta = {k: v for k, v in w.items() if k not in ["id", "text", "importance"]} # will be pattern, description, questions, freq
        weaknesses.append(
            Weakness(id=w_id, text=text, importance=importance, metadata=meta)
        )
    return weaknesses


def _get_course_collection(
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COURSE_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_directory)
    # Important: DO NOT upsert here, just attach to existing index
    collection = client.get_collection(
        name=collection_name,
        embedding_function=get_gemini_embedding_function(),
    )
    return collection


def recommend_courses_for_student(
    weaknesses_raw: List[Dict[str, Any]],
    max_courses_pr_weakness: int = 5,
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COURSE_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Fast online path:
    - assumes course index already built with build_course_index()
    - only queries Chroma; no upsert here
    """

    weaknesses = _parse_weaknesses(weaknesses_raw)

    collection = _get_course_collection(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    all_recommendations: List[CourseScore] = []

    for w in weaknesses:
        print(f"Querying courses for weakness: {w.id} - {w.text[:60]}...")
        result = collection.query(
            query_texts=[w.text],
            n_results=max_courses_pr_weakness,
        )

        # Chroma returns: ids, distances, documents, metadatas
        ids_list = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]  # or "scores" depending on version
        metadatas_list = result.get("metadatas", [[]])[0]

        for cid, d, meta in zip(ids_list, distances, metadatas_list):
            lesson_title = meta.get("lesson_title") or "Untitled course"
            desc = meta.get("description") or ""
            link = meta.get("link") or ""
            course = Course(
                id=str(cid),
                lesson_title=lesson_title,
                description=desc,
                link=link,
                metadata=meta,
            )
            # Convert distance -> similarity (lower distance = higher similarity)
            score = 1 / (1 + float(d))
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
