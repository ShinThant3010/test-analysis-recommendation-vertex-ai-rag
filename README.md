## Test Analysis & Course Recommendation API

FastAPI service that runs the multi-agent pipeline to analyze test results, extract weaknesses, and return course recommendations backed by Vertex AI Matching Engine.

### Prerequisites
- Python 3.11
- Google credentials with access to Vertex AI (Matching Engine + Generative AI)
- Environment variables: `GOOGLE_API_KEY`, optional `API_BEARER_TOKEN` for simple bearer auth

### Installation
```bash
pip install -r requirements.txt
```

### Run locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Health check: `GET /health`  
Pipeline: `POST /api/v1/test-analysis-recommendation`

### Required headers
- `Content-Type: application/json`
- `X-API-Version: 1` (defaults to 1 if omitted)
- `X-Correlation-Id` (auto-generated if omitted; echoed back)
- `Authorization: Bearer <API_BEARER_TOKEN>` (only if you set the env var)

### Request body
```json
{
  "test_id": "5JQC42EJ5E6RHXQAQPDH4AFAXR",
  "student_id": "E1CTEWH0AVNH9DN65R6PPG2X7R",
  "max_courses": 5
}
```

### Response shape (happy path)
```json
{
  "correlation_id": "corr_...",
  "data": {
    "status": "ok",
    "agent1_output": { ... },
    "agent2_output": { ... },
    "weaknesses_llm": [ ... ],
    "course_recommendation": { ... },
    "final_response": { ... }
  }
}
```

### HTTP status mapping (high level)
- `200 OK` pipeline completed (even with warnings)
- `400` invalid API version or bad input
- `401` missing/invalid Authorization (when enabled)
- `404` missing upstream resource (student/test/question/answer)
- `409` duplicate correlation id in-flight
- `500` unexpected pipeline failure
- `502` upstream dependency unavailable (Vertex/Gemini)

### Docker
Build and run:
```bash
docker build -t course-reco-api:latest .
docker run --rm -p 8080:8000 -e GOOGLE_API_KEY=... course-reco-api:latest
```

### Notes
- Vertex index/endpoint must already be deployed; configuration values are read from `config.py`.
- `X-Correlation-Id` is echoed in all responses for tracing.
