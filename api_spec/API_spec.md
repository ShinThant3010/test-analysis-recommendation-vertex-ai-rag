# Test Analysis & Recommendation API — REST API Specification

Service: **test_analysis_recommendation_api (BFF)**

Purpose: Run the multi-agent test-analysis pipeline end-to-end so that downstream clients receive:
1. validated test context,
2. LLM-derived weakness clusters,
3. Vertex AI Matching Engine course recommendations, and
4. a concise user-facing summary.

---

## Base URLs

**Production (Cloud Run):**
`https://test-analysis-recommendation-<project>.run.app`

**Staging (Cloud Run):**
`https://test-analysis-recommendation-<env>.run.app`

**Local (Uvicorn/FastAPI):**
`http://127.0.0.1:8000`

**Swagger / OpenAPI:**
`https://test-analysis-recommendation-service-810737581373.asia-southeast1.run.app/docs`

---

## Guideline Alignment Notes

* ✅ **Resource-based URL:** canonical noun is `/api/v1/test-analyses`
* ✅ **HTTP methods:** `GET` for health, `POST` for synchronous analysis runs
* ✅ **HTTP status codes:** `200 OK` success, `4xx` validation/not-found, `5xx` pipeline errors
* ✅ **Error format:** `{code,message,subErrors,timestamp,correlationId}`
* ✅ **Correlation ID:** `X-Correlation-Id` passthrough + auto-generation
* ✅ **API Version header:** `X-API-Version` (default `1`)
* ✅ **JSON naming:** responses are **camelCase**; requests accept camelCase **and** snake_case
* ✅ **Idempotent inputs:** pipeline deduplicates by `(testId, studentId)` while still treating each request as a fresh run

---

## Authentication & Authorization

**[In Progress — not enforced yet]**

### External Gateway (Bearer JWT)

`Authorization: Bearer <token>`

### Internal Network (X-API-Key)

`X-API-Key: <internal key>`

### Recommended Gateway → Upstream Header Mapping
**[In Progress — not implemented yet]**
* `X-User-Id: <jwt.sub>`
* `X-User-Name: <jwt.name>`
* `X-User-Email: <jwt.email>`
* `X-User-Roles: <jwt.roles>`

---

## Required Headers

### Content Type

`Content-Type: application/json`

### Correlation ID

* Clients **may provide** `X-Correlation-Id`
* Server autogenerates `corr_<uuid>` if missing
* Response always echoes the same ID (success + error)

### API Version

Header versioning complements `/api/v1/...`.

* Required header: `X-API-Version`
* Supported values: `1`
* Invalid version ⇒ `400 INVALID_FIELD_VALUE`

---

## Endpoints Summary

### Health
* `GET /health` 

### Test Analysis Pipeline
* `POST /api/v1/test-analysis-recommendation` 
---

## 1) Health Endpoints

### GET /health ✅ canonical
**Response:** `200 OK`

```json
{
  "status": "ok",
  "service": "test_analysis_recommendation_api",
  "environment": "prod"
}
```

---

## 2) Run Test Analysis & Course Recommendation (REST)

### POST /api/v1/test-analysis-recommendation

Runs the entire Vertex-AI-powered agent pipeline:

1. **Agent 1 – Test Context**: validates `(testId, studentId)` against ExamResult data and surfaces current + historical attempts.
2. **Agent 2 – Incorrect Questions**: joins question/answer data to extract only incorrect items for the current attempt.
3. **Agent 3 – Weakness Extraction**: Gemini 2.5 Flash summarizes reusable weaknesses & patterns from incorrect cases.
4. **Agent 4 – Course Recommendation**: embeds weaknesses + queries the deployed Matching Engine index (`courses-endpoint`) to retrieve candidate courses.
5. **Agent 5 – User-Facing Response**: Gemini 2.5 Flash composes a concise JSON summary plus narrative recommendations for the learner.

### Status Codes

* `200 OK` — pipeline completed successfully (even if some agents produced warnings, e.g., no prior attempts)
* `400 Bad Request` — request validation or unsupported API version
* `404 Not Found` — upstream resource missing (student_id, test_id, question_id, answer_id)
* `409 Conflict` — duplicate request detected while a prior run with the same correlation ID is still in-flight
* `500 Internal Server Error` — unexpected agent failure
* `502 Bad Gateway` — upstream dependencies (Vertex Matching Engine, Gemini) unavailable

### Request Schema

Accepts both camelCase and snake_case keys (service normalizes to snake internally).

#### Example (snake_case)

```json
{
  "test_id": "5JQC42EJ5E6RHXQAQPDH4AFAXR",
  "student_id": "E1CTEWH0AVNH9DN65R6PPG2X7R",
  "max_courses": 5
}
```

#### Example (camelCase)

```json
{
  "testId": "5JQC42EJ5E6RHXQAQPDH4AFAXR",
  "studentId": "E1CTEWH0AVNH9DN65R6PPG2X7R",
  "maxCourses": 5
}
```

#### Field Definitions

| Field        | Type   | Required | Notes                                                                 |
| ------------ | ------ | -------: | --------------------------------------------------------------------- |
| test_id      | string |        ✅ | Assessment/test identifier (maps to `ExamResult.examContentId`)       |
| student_id   | string |        ✅ | Learner identifier (maps to `ExamResult.userId`)                      |
| max_courses  | int    |        ❌ | Total courses to surface in the final list (default `5`, min 1, max 10) |

### Successful Response

**200 OK**

Headers:
* `X-Correlation-Id: corr_...`
* `X-API-Version: 1`

Body (camelCase):

```json
{
  "status": "ok",
  "analysisId": "ta_01J7YV78VJKB1ZQAJN8S4JQ492",
  "testId": "5JQC42EJ5E6RHXQAQPDH4AFAXR",
  "studentId": "E1CTEWH0AVNH9DN65R6PPG2X7R",
  "weaknesses": [
    {
      "id": "01J7YV6P6F39P6VD3C5JXTMX3F",
      "text": "Struggles to compare two-layer percentage scenarios under time pressure",
      "importance": 0.9,
      "metadata": {
        "patternType": "numeracy",
        "description": "Missed discount-comparison questions that require translating textual conditions into equations.",
        "evidenceQuestionIds": [1575, 1622],
        "frequency": 2
      }
    }
  ],
  "courseRecommendations": [
    {
      "weaknessId": "01J7YV6P6F39P6VD3C5JXTMX3F",
      "score": 0.82,
      "reason": "Retrieved by semantic match to weakness \"Struggles to compare two-layer percentage scenarios...\"",
      "course": {
        "id": "COURSE-1198",
        "lessonTitle": "Applied Percentage Reasoning",
        "description": "Hands-on practice with discount chains, ratio frames, and real-world finance examples.",
        "link": "https://learning.example.org/courses/COURSE-1198",
        "metadata": {
          "difficulty": "intermediate",
          "durationHours": 4
        }
      }
    }
  ],
  "userFacingResponse": {
    "summary (final response will be adjusted to be this part only)": {  
      "Test Name": "Computer Science 101"
      "Current Performance": "You handle most reasoning items well but lose accuracy when multi-step percentage conversions appear.",
      "Area to be Improved": "Focus on translating narrative discount problems into structured steps before computing results.",
      "Recommended Course": [
        "Applied Percentage Reasoning addresses the recurring discount comparison mistakes."
      ]
    },
    "recommendations": [
      {
        "courseId": "COURSE-1198",
        "courseTitle": "Applied Percentage Reasoning",
        "targetWeaknessId": "01J7YV6P6F39P6VD3C5JXTMX3F",
        "explanation": "LLM summary referencing why this course fixes the weakness."
      }
    ]
  },
  "agentOutputs": {
    "agent1": {
      "status": "ok",
      "currentTestResult": {
        "id": "5JQC42EJ5E6RHXQAQPDH4AFAXR#3",
        "examContentId": "5JQC42EJ5E6RHXQAQPDH4AFAXR",
        "userId": "E1CTEWH0AVNH9DN65R6PPG2X7R",
        "attemptNumber": 3,
        "earnedScore": 720,
        "totalScore": 1000,
        "status": "completed",
        "createdAt": "2025-01-17T02:14:48.582Z"
      },
      "historyTestResult": {
        "attemptNumber": 2,
        "earnedScore": 680,
        "totalScore": 1000
      },
      "notes": []
    },
    "agent2": {
      "status": "ok",
      "totalQuestionsInTest": 40,
      "totalIncorrectQuestions": 6,
      "incorrectQuestions": [
        {
          "questionId": 1575,
          "questionText": "เลือกคำตอบที่อธิบายลำดับส่วนลดสองชั้น...",
          "studentAnswers": ["B"],
          "correctAnswers": ["D"],
          "difficulty": "medium"
        }
      ]
    },
    "agent3": {
      "model": "gemini-2.5-flash",
      "weaknessCount": 3,
      "rawWeaknesses": [
        {
          "weakness": "Confuses discount chaining with ratio adjustments",
          "pattern_type": "numeracy",
          "description": "...",
          "evidence_question_ids": [1575],
          "frequency": 1,
          "id": "01J7YVW4JCKE3E095PN4TPMWC2"
        }
      ]
    },
    "agent4": {
      "matchingEngineEndpoint": "projects/.../indexEndpoints/5097450255578824704",
      "recommendationCount": 3,
      "maxCourses": 5
    },
    "agent5": {
      "model": "gemini-2.5-flash",
      "status": "ok"
    }
  },
  "metadata": {
    "correlationId": "corr_b6b7c424a7f148c183bb7c1b2addb4e6",
    "apiVersion": "1",
    "generatedAt": "2025-01-17T02:20:11.842Z",
    "tokens": {
      "agent3": {
        "input": 1452,
        "output": 512
      },
      "agent5": {
        "input": 918,
        "output": 314
      }
    }
  }
}
```

### Curl Example

```bash
curl -X 'POST' \
  'https://test-analysis-recommendation-service-810737581373.asia-southeast1.run.app/test-analysis-recommendation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "test_id": "5JQC42EJ5E6RHXQAQPDH4AFAXR",
  "student_id": "E1CTEWH0AVNH9DN65R6PPG2X7R",
  "max_courses": 5
}'
```

---

## 3) Standard Error Format

All errors follow the common schema (camelCase):

```json
{
  "code": "VALIDATION_FAILED",
  "message": "Validation failed",
  "subErrors": [
    {
      "field": "studentId",
      "errors": [
        {
          "code": "missing",
          "message": "Field required"
        }
      ]
    }
  ],
  "timestamp": 1750672014,
  "correlationId": "corr_abc123def456"
}
```

Response headers always echo:

* `X-Correlation-Id`
* `X-API-Version`

### 3.1 Matching Engine Failure — `502 MATCHING_ENGINE_UNAVAILABLE`

Agent 4 could not reach the deployed Vertex endpoint (`find_neighbors` threw an exception). The service returns weaknesses but no recommendations.

### 3.2 Internal Server Error — `500 INTERNAL_SERVER_ERROR`

Thrown for unexpected exceptions (e.g., Gemini quota, data read failure). Logs include per-agent telemetry for debugging.

---

## 4) Enumerations & Schemas

### 4.1 `status` (top-level)

* `ok` — full pipeline completed
* `agent1_error` — ExamResult lookup failure
* `agent2_error` — incorrect question extraction failure
* `no_incorrect_questions` — Agent 2 returned empty set
* `no_weaknesses` — Agent 3 could not extract weaknesses
* `no_course_recommendations` — Agent 4 produced no neighbors
* `agent5_error` — Gemini summary failure (LLM fallback still returns 200 with `userFacingResponse.summary` derived from `_fallback_summary`)

### 4.2 Token Logging

`metadata.tokens` mirrors entries stored in `token_log.json`:

```json
{
  "usage": "agent3: weakness extraction",
  "inputTokens": 1452,
  "outputTokens": 512,
  "runtimeSeconds": 3.42,
  "timestamp": "2025-01-17T02:20:11.842Z"
}
```

---

## 5) Internal Dependencies

| Component | Purpose | Notes |
| --------- | ------- | ----- |
| `vertexai.MatchingEngineIndexEndpoint` | Retrieves courses semantically similar to weaknesses | Uses `INDEX_ENDPOINT_NAME` + `ENDPOINT_DISPLAY_NAME` from `config.py` |
| `google.genai` (`gemini-2.5-flash`) | Agents 3 & 5 LLM calls | Requires `GOOGLE_API_KEY` |
| `token_logger.py` | Captures token usage + runtime for observability | Appends to `token_log.json` |
| `agents.agent?_*` modules | Each agent is importable & testable independently | API composes them via `pipeline/run_pipeline.py` |

---

## 6) Change Log

* **2025-01-19**: Initial specification drafted.

---
