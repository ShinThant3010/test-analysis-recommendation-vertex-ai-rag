"""
End-to-end helper that prepares Vertex AI Vector Search resources:

1. Load the course CSV defined in config.
2. Embed every row with the configured Gemini embedding model.
3. Write JSONL shards, upload them to GCS, and create/update a Matching Engine index.
4. Create the index endpoint (if necessary), deploy the index, and push the datapoints via upsert.

All parameters are pulled from config.py so the script can run without CLI args.
"""

from __future__ import annotations

import json
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Iterable, List, Sequence
import warnings

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1.types import IndexDatapoint

from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform.matching_engine import matching_engine_index_config
from google.protobuf import struct_pb2
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress Vertex AI model garden deprecation warnings until migration is ready.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="vertexai._model_garden._model_garden_models",
)

from config import (  # noqa: E402
    BATCH_SIZE,
    COURSE_CSV_PATH,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEFAULT_SHARD_SIZE,
    DEPLOYED_INDEX_ID,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    ENDPOINT_DISPLAY_NAME,
    ENDPOINT_MACHINE_TYPE,
    ENDPOINT_MAX_REPLICAS,
    ENDPOINT_MIN_REPLICAS,
    GCS_VECTOR_BUCKET,
    GCS_VECTOR_PREFIX,
    INDEX_DESCRIPTION,
    INDEX_DISPLAY_NAME,
    INDEX_ENDPOINT_NAME,
    LOCAL_VECTOR_OUTPUT_DIR,
    PUBLIC_ENDPOINT_ENABLED,
    TREE_AH_APPROXIMATE_NEIGHBORS_COUNT,
    TREE_AH_LEAF_NODE_EMBEDDING_COUNT,
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
        text = "\n".join(
            f"{column}: {value}" for column, value in row_dict.items() if value not in ("", None)
        )
        docs.append({"id": doc_id, "text": text, "metadata": row_dict})
    return docs


def chunk(sequence: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


@log_call
def embed_texts(model: TextEmbeddingModel, texts: Sequence[str]) -> List[List[float]]:
    embeddings = model.get_embeddings(texts)
    return [list(embedding.values) for embedding in embeddings]


@log_call
def build_json_records(docs: Sequence[dict], embeddings: Sequence[Sequence[float]]) -> List[dict]:
    records = []
    for doc, emb in zip(docs, embeddings):
        records.append(
            {
                "id": doc["id"],
                "embedding": list(emb),
                "restricts": [{"namespace": "type", "allow": ["course"]}],
                "metadata": {"raw_text": doc["text"]},
            }
        )
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


def gcs_uri(bucket: str, prefix: str) -> str:
    normalized_prefix = prefix.strip().strip("/")
    return f"gs://{bucket}/{normalized_prefix}/" if normalized_prefix else f"gs://{bucket}/"


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


@log_call
def ensure_index(contents_uri: str) -> MatchingEngineIndex:
    """Reuse an existing Matching Engine index or create a new one."""
    indexes = MatchingEngineIndex.list(
        project=DEFAULT_PROJECT_ID,
        location=DEFAULT_LOCATION,
    )
    for idx in indexes:
        if idx.display_name == INDEX_DISPLAY_NAME:
            print(f"Reusing: {idx.resource_name}")
            return idx
    
    print("Index not found. Creating a new one...")
    index = MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        description=INDEX_DESCRIPTION,
        dimensions=EMBEDDING_DIMENSION,
        distance_measure_type=matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        contents_delta_uri=contents_uri,
        leaf_node_embedding_count=TREE_AH_LEAF_NODE_EMBEDDING_COUNT,
        approximate_neighbors_count=TREE_AH_APPROXIMATE_NEIGHBORS_COUNT,
        project=DEFAULT_PROJECT_ID,
        location=DEFAULT_LOCATION,
        index_update_method="STREAM_UPDATE",
    )
    index.wait()
    print(f"Created index: {index.resource_name}")
    return index

@log_call
def ensure_endpoint() -> MatchingEngineIndexEndpoint:
    """Reuse an existing Matching Engine index Endpoint or create a new one."""
    endpoints = MatchingEngineIndexEndpoint.list(
        project=DEFAULT_PROJECT_ID, 
        location=DEFAULT_LOCATION,
    )
    for ep in endpoints:
        if ep.display_name == ENDPOINT_DISPLAY_NAME:
            print(f"Reusing existing endpoint: {ep.resource_name}")
            return ep
        
    print("Endpoint not found. Creating a new one...")
    endpoint = MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        project=DEFAULT_PROJECT_ID,
        location=DEFAULT_LOCATION,
        public_endpoint_enabled=PUBLIC_ENDPOINT_ENABLED,
    )
    endpoint.wait()
    print(f"Created endpoint: {endpoint.resource_name}")
    return endpoint


@log_call
def ensure_deployment(
    endpoint: MatchingEngineIndexEndpoint,
    index: MatchingEngineIndex,
) -> None:
    deployed_indexes = getattr(endpoint.gca_resource, "deployed_indexes", []) or []
    for deployed in deployed_indexes:
        if getattr(deployed, "id", "") == DEPLOYED_INDEX_ID:
            print(f"Index already deployed as {DEPLOYED_INDEX_ID}")
            return

    endpoint.deploy_index(
        index=index,
        deployed_index_id=DEPLOYED_INDEX_ID,
        display_name=ENDPOINT_DISPLAY_NAME,
        machine_type=ENDPOINT_MACHINE_TYPE,
        min_replica_count=ENDPOINT_MIN_REPLICAS,
        max_replica_count=ENDPOINT_MAX_REPLICAS,
        sync=True,
    )
    print(f"Deployed index {index.resource_name} to endpoint {endpoint.resource_name}")


def _clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


@log_call
def build_datapoints(docs: Sequence[dict], embeddings: Sequence[Sequence[float]]):
    datapoints: List[IndexDatapoint] = []
    for doc, embedding in zip(docs, embeddings):
        metadata_struct = struct_pb2.Struct()
        metadata_struct.update(_clean_metadata(doc["metadata"]))
        datapoint = IndexDatapoint(
            datapoint_id=doc["id"],
            feature_vector=list(embedding),
            restricts=[
                IndexDatapoint.Restriction(
                    namespace="type", allow_list=["course"]
                )
            ],
            embedding_metadata=metadata_struct,
        )
        datapoints.append(datapoint)
    return datapoints


@log_call
def upsert_datapoints(index: MatchingEngineIndex, datapoints) -> None:
    index.upsert_datapoints(datapoints=datapoints)
    print(f"Upserted {len(datapoints)} datapoints into {index.resource_name}")


def main() -> None:
    if not COURSE_CSV_PATH.exists():
        raise SystemExit(f"CSV file not found: {COURSE_CSV_PATH}")

    vertex_init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
    aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)

    docs = load_documents(COURSE_CSV_PATH)
    if not docs:
        raise SystemExit("No rows found in the CSV.")

    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

    embeddings: List[List[float]] = []
    for batch in chunk(docs, BATCH_SIZE):
        embeddings.extend(embed_texts(embedding_model, [doc["text"] for doc in batch]))

    records = build_json_records(docs, embeddings)
    shard_paths = write_shards(records, LOCAL_VECTOR_OUTPUT_DIR, DEFAULT_SHARD_SIZE)
    upload_to_gcs(DEFAULT_PROJECT_ID, GCS_VECTOR_BUCKET, GCS_VECTOR_PREFIX, shard_paths)
    contents_uri = gcs_uri(GCS_VECTOR_BUCKET, GCS_VECTOR_PREFIX)

    index = ensure_index(contents_uri)
    endpoint = ensure_endpoint()
    ensure_deployment(endpoint, index)

    # datapoints = build_datapoints(docs, embeddings)
    # upsert_datapoints(index, datapoints)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
