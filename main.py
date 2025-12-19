from typing import Any, Dict

from fastapi import FastAPI, HTTPException
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


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.post("/test-analysis-recommendation")
def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    """
    Execute the LLM pipeline using the supplied parameters.
    """
    try:
        result = run_full_pipeline(
            test_id=request.test_id,
            student_id=request.student_id,
            max_courses=request.max_courses,
        )
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run pipeline: {exc}",
        ) from exc
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
