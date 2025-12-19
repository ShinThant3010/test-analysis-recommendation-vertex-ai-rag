# config.py
import os
from google import genai
from dotenv import load_dotenv

from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

load_dotenv()

MAX_COURSES = 5

# ==== CSV PATHS ====
dataset = "_data/data_cs"
TEST_ID = "5JQC42EJ5E6RHXQAQPDH4AFAXR"
STUDENT_ID = "E1CTEWH0AVNH9DN65R6PPG2X7R"

# dataset = "_data/data_general_logical"
# TEST_ID = "5JQC42EJ5E6RHXQAQPDH4AFAXR"
# STUDENT_ID = "E1CTEWH0AVNH9DN65R6PPG2X7R"

# dataset = "_data/data_general_procedual"
# TEST_ID = "5JQC42EJ5E6RHXQAQPDH4AFAXR"
# STUDENT_ID = "E1CTEWH0AVNH9DN65R6PPG2X7R"

# dataset = "_data/data_toeic"
# TEST_ID = "5JQC42EJ5E6RHXQAQPDH4AFAXR"
# STUDENT_ID = "E1CTEWH0AVNH9DN65R6PPG2X7R"

QUESTION_PATH = dataset + "/Question.csv"
ANSWER_PATH   = dataset + "/Answer.csv"
TQ_PATH       = dataset + "/ExamQuestionResult.csv"
TA_PATH       = dataset + "/ExamAnswerResult.csv"
TEST_RESULT_PATH = dataset + "/ExamResult.csv"
COURSE_PATH      = "_data/course/course.csv"
TOKEN_LOG_PATH = os.getenv("TOKEN_LOG_PATH", "token_log.json")

# ==== Generation Model ====
GENERATION_MODEL = "gemini-2.5-flash"

# ==== Chroma defaults (values can be overridden via environment variables) ====
PERSIST_DIRECTORY = os.getenv("COURSE_CHROMA_DIR", "chroma_courses")
COURSE_COLLECTION_NAME = os.getenv("COURSE_COLLECTION_NAME", "courses")

# ==== Gemini / Embeddings ====
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")
client = genai.Client(api_key=API_KEY)

# Default model names
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


# ====== Data models (simplified) ======
@dataclass
class Course:
    id: str
    lesson_title: str
    description: str
    link: str
    metadata: Dict[str, Any] | None = None

@dataclass
class Weakness:
    id: str
    text: str
    importance: float = 1.0
    metadata: Dict[str, Any] = None # the rest

@dataclass
class CourseScore:
    course: Course
    weakness_id: str
    score: float
    reason: str


# Default BigQuery target
DEFAULT_PROJECT_ID = "poc-piloturl-nonprod"
DEFAULT_DATASET_ID = "dev_learning_service"
DEFAULT_TABLE_ID = "dbo_Course"

# Default number of rows to fetch/print if --max-rows is not provided.
DEFAULT_MAX_ROWS = 10

# Where CSV exports should be written.
OUTPUT_DIR = Path("_data") / "bigdata_query"

# Local course CSV used for prototyping RAG flows.
COURSE_CSV_PATH = Path("_data") / "courses" / "course.csv"

# Vertex AI defaults
DEFAULT_LOCATION = "asia-southeast1"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
GENERATION_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_RETRIEVAL_K = 3
EMBEDDING_DIMENSION = 3072

# Cloud Storage staging for Matching Engine ingestion
LOCAL_VECTOR_OUTPUT_DIR = Path("_data") / "vector_upload"
GCS_VECTOR_BUCKET = "poc-piloturl-test-analysis-rag-data"
GCS_VECTOR_PREFIX = "vectors/courses"
DEFAULT_SHARD_SIZE = 100  # documents per JSONL shard
BATCH_SIZE = 32  # embedding batch size

# Vertex AI Matching Engine index defaults
INDEX_NAME = "courses-index"
INDEX_DISPLAY_NAME = "Courses Index"
INDEX_DESCRIPTION = "Course embeddings ingested from course.csv"
INDEX_ENDPOINT_NAME = "courses-endpoint"
DEFAULT_MATCHING_ALGORITHM = "tree-ah"
TREE_AH_LEAF_NODE_EMBEDDING_COUNT = 1000
TREE_AH_APPROXIMATE_NEIGHBORS_COUNT = 10
DEPLOYED_INDEX_ID = "courses_deployment"
ENDPOINT_DISPLAY_NAME = "Courses Endpoint"
ENDPOINT_MACHINE_TYPE = "e2-standard-16"
ENDPOINT_MIN_REPLICAS = 1
ENDPOINT_MAX_REPLICAS = 1
DEFAULT_UPDATE_ACTION = "store_true"
PUBLIC_ENDPOINT_ENABLED = True  # TODO: revisit with the team if private networking is required.