"""
Create or update a Vertex AI Matching Engine index from staged Cloud Storage data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from google.cloud import aiplatform
import time
from functools import wraps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DISTANCE_MEASURE,
    EMBEDDING_DIMENSION,
    GCS_VECTOR_BUCKET,
    GCS_VECTOR_PREFIX,
    INDEX_DESCRIPTION,
    INDEX_DISPLAY_NAME,
    DEFAULT_MATCHING_ALGORITHM,
    TREE_AH_LEAF_NODE_EMBEDDING_COUNT,
    TREE_AH_APPROXIMATE_NEIGHBORS_COUNT,
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
def gcs_uri(bucket: str, prefix: str) -> str:
    normalized_prefix = prefix.strip("/")
    return f"gs://{bucket}/{normalized_prefix}/" if normalized_prefix else f"gs://{bucket}/"

@log_call
def create_index() -> aiplatform.MatchingEngineIndex:
    # Create a new Matching Engine index from GCS staged data.
    contents_uri = gcs_uri(GCS_VECTOR_BUCKET, GCS_VECTOR_PREFIX)

    # Create the index
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        description=INDEX_DESCRIPTION,
        dimensions=EMBEDDING_DIMENSION,
        distance_measure_type=aiplatform.matching_engine.matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        contents_delta_uri=contents_uri,
        leaf_node_embedding_count=TREE_AH_LEAF_NODE_EMBEDDING_COUNT,
        approximate_neighbors_count=TREE_AH_APPROXIMATE_NEIGHBORS_COUNT,
        project=DEFAULT_PROJECT_ID,
        location=DEFAULT_LOCATION,
    )
    index.wait()
    return index


def main() -> None:
    aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)

    index = create_index()
    print(f"Created index {index.resource_name} ({DEFAULT_MATCHING_ALGORITHM}) from {gcs_uri(GCS_VECTOR_BUCKET, GCS_VECTOR_PREFIX)}")


if __name__ == "__main__":
    main()
