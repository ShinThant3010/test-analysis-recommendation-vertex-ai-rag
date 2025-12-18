import os
import time
from functools import wraps
from typing import Dict, List
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
from google import genai
from google.genai.types import EmbedContentConfig
import vertexai
from vertexai.generative_models import GenerativeModel

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    COURSE_CSV_PATH,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEPLOYED_INDEX_ID,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
)

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

vertexai.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
genai_client = genai.Client()

def log_call(func):
    """Decorator that reports runtime for each function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f"[Runtime] Calling {func.__name__}")
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[Runtime] {func.__name__} finished in {elapsed:.2f}s")
    return wrapper

# ---------------------------
# 1) Embed helpers (Gemini 001)
# ---------------------------
@log_call
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

# ---------------------------
# 4) Query (embed query → nearest neighbors)
# ---------------------------
@log_call
def nearest_neighbors(endpoint_name: str, query: str, k: int = 5):
    print(f"Querying Vector Search for: {query}")
    qvec = embed_texts([query])[0]
    ep = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
    
    # find_neighbors is the public kNN call on the endpoint
    res = ep.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[qvec],
        num_neighbors=k,
        return_full_datapoint=True
    )
    # Save result
    with open("result.txt", 'w') as f:
        f.write(res.__str__())
    print(f"Saved result to result.txt")
    
    return res

# ---------------------------
# 5) Generate with Gemini 2.5 using retrieved context
# ---------------------------
@log_call
def answer_with_gemini(context_snippets: List[str], question: str) -> str:
    model = GenerativeModel("gemini-2.5-flash") 
    context = "\n\n".join(context_snippets)
    prompt = f"""Use only the context to answer. Provide a detailed and comprehensive response with thorough explanations. Include all relevant information from the context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}
Answer (provide a detailed explanation with at least 3-4 paragraphs):"""
    return model.generate_content(prompt).text

@log_call
def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    print("Chunking text...")
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len and text[end] != ' ':
            # Find the last space within the chunk to avoid cutting words
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def load_course_texts(csv_path: Path) -> Dict[str, str]:
    import csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Course CSV not found: {csv_path}")

    course_map: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            course_id = row.get("id")
            if not course_id:
                continue
            parts = [
                row.get("lesson_title", ""),
                row.get("short_description", ""),
                row.get("description", ""),
            ]
            text = "\n".join(filter(None, parts)).strip()
            course_map[course_id] = text
    return course_map

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    endpoint_name = "projects/810737581373/locations/asia-southeast1/indexEndpoints/5097450255578824704"
    index_name = "projects/810737581373/locations/asia-southeast1/indexes/3402319984897032192"
    try:
        course_lookup = load_course_texts(COURSE_CSV_PATH)
        # Step 3: Test query
        query = "What courses should I take to improve my data analysis skills?"
        print(f"\nTesting query: {query}")
        
        # Step 4: Retrieve relevant chunks
        neighbors = nearest_neighbors(endpoint_name, query, k=3)
        retrieved_chunks = []

        neighbor_ids = [n.id for n in neighbors[0]]
        # → use these IDs to fetch course/content records from DB

        print("\nRetrieved documents:")
        for i, neighbor in enumerate(neighbors[0]):
            content = course_lookup.get(neighbor.id)
            if content:
                retrieved_chunks.append(content)

            preview = (content or "")[:150]
            if content and len(content) > 150:
                preview += "..."
            print(f"Document {i+1} - ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
            print(f"Content: {preview}\n")
        
        # Step 5: Generate answer
        # print("\nGenerating answer...")
        # answer = answer_with_gemini(retrieved_chunks, query)
        # print(f"Answer: {answer}")
        
        print("\nVector Search index created and tested successfully!")
        print(f"Index name: {index_name}")
        print(f"Endpoint name: {endpoint_name}")
        
    except Exception as e:
        print(f"Error: {e}")
