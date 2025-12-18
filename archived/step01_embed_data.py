"""
Create Vertex AI Vector Search ingestion files from the local course CSV and upload them to GCS.
"""

from __future__ import annotations

import time
from functools import wraps
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from google.cloud import storage
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    COURSE_CSV_PATH,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEFAULT_SHARD_SIZE,
    BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    GCS_VECTOR_BUCKET,
    GCS_VECTOR_PREFIX,
    LOCAL_VECTOR_OUTPUT_DIR,
)

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

@log_call
def load_documents(csv_path: Path) -> List[dict]:
    df = pd.read_csv(csv_path).fillna("")
    docs: List[dict] = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        doc_id = str(row_dict.get("id", idx))
        text = "\n".join(f"{k}: {v}" for k, v in row_dict.items() if v not in ("", None))
        docs.append({"id": doc_id, "text": text, "metadata": row_dict})
    return docs

@log_call
def chunk(sequence: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]

@log_call
def embed_texts(model: TextEmbeddingModel, texts: Sequence[str]) -> List[List[float]]:
    embeddings = model.get_embeddings(texts)
    return [emb.values for emb in embeddings]

@log_call
def build_json_records(docs: Sequence[dict], embeddings: Sequence[Sequence[float]]) -> List[dict]:
    records = []
    for doc, emb in zip(docs, embeddings):
        record = {
            "id": doc["id"],
            "embedding": list(emb),
            "restricts": [{"namespace": "type", "allow": ["course"]}],
            "metadata": {
                "raw_text": doc["text"],
            },
        }
        records.append(record)
    return records

@log_call
def write_shards(records: Sequence[dict], output_dir: Path, shard_size: int) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []
    shard_index = 0
    for start in range(0, len(records), shard_size):
        shard_index += 1
        shard_records = records[start : start + shard_size]
        shard_path = output_dir / f"vectors-shard-{shard_index:04d}.json"
        with shard_path.open("w", encoding="utf-8") as handle:
            for record in shard_records:
                json.dump(record, handle)
                handle.write("\n")
        shard_paths.append(shard_path)
    return shard_paths

@log_call
def upload_to_gcs(project_id: str, bucket_name: str, prefix: str, files: Sequence[Path]) -> List[str]:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    uploaded: List[str] = []
    normalized_prefix = prefix.strip().strip("/")
    for file_path in files:
        blob_name = f"{normalized_prefix}/{file_path.name}" if normalized_prefix else file_path.name
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        uploaded.append(f"gs://{bucket_name}/{blob_name}")
    return uploaded


def main() -> None:
    shard_paths: List[Path] = []

    if not COURSE_CSV_PATH.exists():
        raise SystemExit(f"CSV file not found: {COURSE_CSV_PATH}")

    docs = load_documents(COURSE_CSV_PATH)
    if not docs:
        raise SystemExit("CSV file is empty.")

    vertex_init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(docs)} documents using {EMBEDDING_MODEL_NAME}...")
    embeddings: List[List[float]] = []
    for batch_docs in chunk(docs, BATCH_SIZE):
        batch_embeddings = embed_texts(embedding_model, [doc["text"] for doc in batch_docs])
        embeddings.extend(batch_embeddings)

    print("Building JSONL records...")
    records = build_json_records(docs, embeddings)
    shard_paths = write_shards(records, LOCAL_VECTOR_OUTPUT_DIR, DEFAULT_SHARD_SIZE)
    print(f"Wrote {len(shard_paths)} shard(s) to {LOCAL_VECTOR_OUTPUT_DIR}")

    print(f"Uploading shards to gs://{GCS_VECTOR_BUCKET}/{GCS_VECTOR_PREFIX} ...")
    uploaded = upload_to_gcs(DEFAULT_PROJECT_ID, GCS_VECTOR_BUCKET, GCS_VECTOR_PREFIX, shard_paths)
    for uri in uploaded:
        print(f"Uploaded {uri}")


if __name__ == "__main__":
    main()
