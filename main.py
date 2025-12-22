from typing import Any, Dict
import os
import uuid
import threading

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field
from google.api_core.exceptions import GoogleAPIError

from config import (
    MAX_COURSES,
    STUDENT_ID,
    TEST_ID,
)
from pipeline.run_pipeline import run_full_pipeline

_active_correlation_ids: set[str] = set()
_corr_lock = threading.Lock()
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")


class PipelineRequest(BaseModel):
    test_id: str = Field(default=TEST_ID, description="Exam/test content identifier.")
    student_id: str = Field(default=STUDENT_ID, description="Student identifier.")
    max_courses: int = Field(
        default=MAX_COURSES,
        ge=1,
        description="Maximum courses retrieved per weakness.",
    )


app = FastAPI(
    title="Test Analysis & Course Recommendation API",
    version="0.1.0",
    description="Run the analysis pipeline via HTTP endpoints.",
)
router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])


def require_headers(
    response: Response,
    x_api_version: str | None = Header(None, alias="X-API-Version", include_in_schema=False),
    x_correlation_id: str | None = Header(None, alias="X-Correlation-Id", include_in_schema=False),
    content_type: str | None = Header(None, alias="Content-Type", include_in_schema=False),
    authorization: str | None = Header(None, alias="Authorization", include_in_schema=False),
) -> Dict[str, str]:
    """
    Enforce required API headers and propagate correlation id.
    """
    correlation_id = x_correlation_id or f"corr_{uuid.uuid4()}"
    response.headers["X-Correlation-Id"] = correlation_id
    version = x_api_version or "1"
    response.headers["X-API-Version"] = version

    if version != "1":
        detail = {
            "code": "INVALID_FIELD_VALUE",
            "message": f"Unsupported X-API-Version: {version}",
            "correlation_id": correlation_id,
        }
        raise HTTPException(
            status_code=400,
            detail=detail,
            headers={"X-Correlation-Id": correlation_id, "X-API-Version": version},
        )

    if content_type and not content_type.lower().startswith("application/json"):
        detail = {
            "code": "INVALID_CONTENT_TYPE",
            "message": f"Content-Type must be application/json, got: {content_type}",
            "correlation_id": correlation_id,
        }
        raise HTTPException(
            status_code=415,
            detail=detail,
            headers={"X-Correlation-Id": correlation_id, "X-API-Version": version},
        )

    if API_BEARER_TOKEN:
        expected = f"Bearer {API_BEARER_TOKEN}"
        if authorization != expected:
            detail = {
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing Authorization header.",
                "correlation_id": correlation_id,
            }
            raise HTTPException(
                status_code=401,
                detail=detail,
                headers={"X-Correlation-Id": correlation_id, "X-API-Version": version},
            )

    return {"correlation_id": correlation_id}


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health-check endpoint."""
    health_check = {
        "status": "ok",
        "service": "test_analysis_recommendation_api",
        "environment": "prod"
        }
    return health_check


@router_v1.post("/test-analysis-recommendation")
def run_pipeline_v1(
    request: PipelineRequest,
    response: Response,
    context: Dict[str, str] = Depends(require_headers),
) -> Dict[str, Any]:
    """
    Execute the LLM pipeline using the supplied parameters.
    """
    correlation_id = context["correlation_id"]
    # Enforce idempotency guard: reject duplicates while in-flight.
    with _corr_lock:
        if correlation_id in _active_correlation_ids:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "CONFLICT",
                    "message": "Request with this correlation id is already processing.",
                    "correlation_id": correlation_id,
                },
                headers={"X-Correlation-Id": correlation_id, "X-API-Version": response.headers.get("X-API-Version", "1")},
            )
        _active_correlation_ids.add(correlation_id)

    status_code = 200
    try:
        result = run_full_pipeline(
            test_id=request.test_id,
            student_id=request.student_id,
            max_courses=request.max_courses,
        )
        # Map known pipeline statuses to HTTP codes.
        status = result.get("status")
        if status == "agent1_error":
            status_code = 404
        elif status == "no_course_recommendations":
            status_code = 200
        elif status == "agent2_error":
            status_code = 404
        elif status == "no_incorrect_questions":
            status_code = 200
        elif status == "no_weaknesses":
            status_code = 200
        else:
            status_code = 200

    except HTTPException:
        # Pass through pre-built HTTP exceptions.
        with _corr_lock:
            _active_correlation_ids.discard(correlation_id)
        raise
    except GoogleAPIError as exc:
        with _corr_lock:
            _active_correlation_ids.discard(correlation_id)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "UPSTREAM_UNAVAILABLE",
                "message": f"Upstream dependency unavailable: {exc}",
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-Id": correlation_id, "X-API-Version": response.headers.get("X-API-Version", "1")},
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guardrail
        with _corr_lock:
            _active_correlation_ids.discard(correlation_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"Failed to run pipeline: {exc}",
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-Id": correlation_id, "X-API-Version": response.headers.get("X-API-Version", "1")},
        ) from exc
    finally:
        with _corr_lock:
            _active_correlation_ids.discard(correlation_id)

    response.status_code = status_code
    return {"correlation_id": correlation_id, "data": result}


app.include_router(router_v1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
