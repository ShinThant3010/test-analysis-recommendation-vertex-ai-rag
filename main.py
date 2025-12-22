from typing import Any, Dict
import uuid

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from config import (
    MAX_COURSES,
    STUDENT_ID,
    TEST_ID,
)
from pipeline.run_pipeline import run_full_pipeline


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
    x_api_version: str = Header(..., alias="X-API-Version"),
    x_correlation_id: str | None = Header(None, alias="X-Correlation-Id"),
) -> Dict[str, str]:
    """
    Enforce required API headers and propagate correlation id.
    """
    correlation_id = x_correlation_id or f"corr_{uuid.uuid4()}"
    response.headers["X-Correlation-Id"] = correlation_id

    if x_api_version != "1":
        detail = {
            "code": "INVALID_FIELD_VALUE",
            "message": f"Unsupported X-API-Version: {x_api_version}",
            "correlation_id": correlation_id,
        }
        raise HTTPException(
            status_code=400,
            detail=detail,
            headers={"X-Correlation-Id": correlation_id},
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
    context: Dict[str, str] = Depends(require_headers),
) -> Dict[str, Any]:
    """
    Execute the LLM pipeline using the supplied parameters.
    """
    correlation_id = context["correlation_id"]
    try:
        result = run_full_pipeline(
            test_id=request.test_id,
            student_id=request.student_id,
            max_courses=request.max_courses,
        )
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"Failed to run pipeline: {exc}",
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-Id": correlation_id},
        ) from exc
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
