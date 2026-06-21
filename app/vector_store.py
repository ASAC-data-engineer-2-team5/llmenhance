from uuid import UUID

from qdrant_client import QdrantClient, models

REQUIRED_PAYLOAD_FIELDS = ("chunk_id", "document_id", "source_path", "title")
MAX_QDRANT_UNSIGNED_INTEGER_ID = 2**64 - 1

# named vector 이름 — dense(bge-m3)와 sparse(BM25)를 한 컬렉션에 함께 둔다.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

SEARCH_MODES = ("dense", "sparse", "hybrid")


def ensure_collection(qdrant_url: str, collection_name: str, vector_size: int) -> None:
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than 0")

    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        if _has_required_hybrid_schema(collection):
            return
        client.delete_collection(collection_name=collection_name)

    _create_collection(client, collection_name, vector_size)


def _create_collection(client, collection_name: str, vector_size: int) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            # IDF modifier 를 켜면 BM25 의 IDF 항을 Qdrant 가 컬렉션 통계로 계산한다.
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def delete_collection_if_exists(qdrant_url: str, collection_name: str) -> None:
    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)


def upsert_chunk_vectors(
    qdrant_url: str,
    collection_name: str,
    points: list[dict],
) -> None:
    if not points:
        return

    client = QdrantClient(url=qdrant_url)
    qdrant_points = []
    for point in points:
        _validate_point(point)
        sparse = point["sparse"]
        qdrant_points.append(
            models.PointStruct(
                id=point["id"],
                vector={
                    DENSE_VECTOR_NAME: point["dense"],
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                },
                payload=point["payload"],
            )
        )

    client.upsert(collection_name=collection_name, points=qdrant_points)


def search_chunks(
    qdrant_url: str,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector: dict[str, list] | None,
    top_k: int,
    metadata_filter: dict[str, str] | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    """dense(bge-m3) + sparse(BM25) 검색을 RRF 로 결합해 반환한다.

    mode="dense"/"sparse" 는 실험 비교용으로 한쪽 벡터만 사용한다(둘 다 RRF 단일 prefetch).
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not dense_vector:
        raise ValueError("dense_vector must not be empty")

    query_filter = _build_filter(metadata_filter)
    sparse_indices, sparse_values = _validate_sparse_query_vector(sparse_vector)

    if mode == "sparse" and not sparse_indices:
        return []

    prefetch = []
    if mode in ("dense", "hybrid"):
        prefetch.append(
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=top_k,
                filter=query_filter,
            )
        )
    if mode in ("sparse", "hybrid") and sparse_indices:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=SPARSE_VECTOR_NAME,
                limit=top_k,
                filter=query_filter,
            )
        )

    client = QdrantClient(url=qdrant_url)
    response = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "id": point.id,
            "score": point.score,
            "payload": point.payload or {},
        }
        for point in response.points
    ]


def _has_required_hybrid_schema(collection) -> bool:
    params = getattr(getattr(collection, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)
    if sparse_vectors is None:
        sparse_vectors = getattr(params, "sparse_vectors_config", None)

    return _has_named_vector(vectors, DENSE_VECTOR_NAME) and _has_named_vector(
        sparse_vectors, SPARSE_VECTOR_NAME
    )


def _has_named_vector(vector_config, vector_name: str) -> bool:
    if isinstance(vector_config, dict):
        return vector_name in vector_config
    return hasattr(vector_config, vector_name)


def _build_filter(metadata_filter: dict[str, str] | None):
    if not metadata_filter:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=key, match=models.MatchValue(value=_coerce_filter_value(value))
            )
            for key, value in metadata_filter.items()
        ]
    )


def _coerce_filter_value(value):
    """CLI 필터는 문자열로 들어오지만 jo_no·hang_no 같은 payload 필드는 정수다.
    정수로 변환 가능한 문자열은 int 로 맞춰 타입 불일치를 막는다."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _validate_sparse_query_vector(
    sparse_vector: dict[str, list] | None,
) -> tuple[list, list]:
    if not sparse_vector:
        return [], []
    if not isinstance(sparse_vector, dict):
        raise ValueError("sparse_vector must be a dict with 'indices' and 'values'")

    indices = sparse_vector.get("indices", [])
    values = sparse_vector.get("values", [])
    if len(indices) != len(values):
        raise ValueError("sparse_vector 'indices' and 'values' must be the same length")
    return indices, values


def _validate_point(point: dict) -> None:
    for field_name in ("id", "dense", "sparse", "payload"):
        if field_name not in point:
            raise ValueError(f"point is missing required field: {field_name}")

    _validate_point_id(point["id"])

    if not point["dense"]:
        raise ValueError("point dense vector must not be empty")

    sparse = point["sparse"]
    if not isinstance(sparse, dict) or "indices" not in sparse or "values" not in sparse:
        raise ValueError("point sparse must be a dict with 'indices' and 'values'")
    if len(sparse["indices"]) != len(sparse["values"]):
        raise ValueError("point sparse 'indices' and 'values' must be the same length")

    payload = point["payload"]
    if not isinstance(payload, dict):
        raise ValueError("point payload must be a dict")

    for field_name in REQUIRED_PAYLOAD_FIELDS:
        if field_name not in payload:
            raise ValueError(f"point payload is missing required field: {field_name}")
        if not isinstance(payload[field_name], str) or not payload[field_name].strip():
            raise ValueError(f"point payload field must be a non-empty string: {field_name}")


def _validate_point_id(point_id) -> None:
    if isinstance(point_id, bool):
        raise ValueError("point id must be an unsigned integer or UUID string")

    if isinstance(point_id, int):
        if 0 <= point_id <= MAX_QDRANT_UNSIGNED_INTEGER_ID:
            return
        raise ValueError("point id must be an unsigned integer or UUID string")

    if isinstance(point_id, str):
        try:
            UUID(point_id)
        except ValueError as exc:
            raise ValueError("point id must be an unsigned integer or UUID string") from exc
        return

    raise ValueError("point id must be an unsigned integer or UUID string")
