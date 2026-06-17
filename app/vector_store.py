from uuid import UUID

from qdrant_client import QdrantClient, models


REQUIRED_PAYLOAD_FIELDS = ("chunk_id", "document_id", "source_path", "title")
MAX_QDRANT_UNSIGNED_INTEGER_ID = 2**64 - 1


def ensure_collection(qdrant_url: str, collection_name: str, vector_size: int) -> None:
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than 0")

    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


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
        qdrant_points.append(
            models.PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"],
            )
        )

    client.upsert(collection_name=collection_name, points=qdrant_points)


def search_chunks(
    qdrant_url: str,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    candidate_chunk_ids: list[str] | None = None,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not query_vector:
        raise ValueError("query_vector must not be empty")
    if candidate_chunk_ids == []:
        return []

    query_filter = None
    if candidate_chunk_ids is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=candidate_chunk_ids),
                )
            ]
        )

    client = QdrantClient(url=qdrant_url)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
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


def _validate_point(point: dict) -> None:
    for field_name in ("id", "vector", "payload"):
        if field_name not in point:
            raise ValueError(f"point is missing required field: {field_name}")

    _validate_point_id(point["id"])

    payload = point["payload"]
    if not isinstance(payload, dict):
        raise ValueError("point payload must be a dict")

    for field_name in REQUIRED_PAYLOAD_FIELDS:
        if field_name not in payload:
            raise ValueError(f"point payload is missing required field: {field_name}")
        if not isinstance(payload[field_name], str) or not payload[field_name].strip():
            raise ValueError(
                f"point payload field must be a non-empty string: {field_name}"
            )


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
            raise ValueError(
                "point id must be an unsigned integer or UUID string"
            ) from exc
        return

    raise ValueError("point id must be an unsigned integer or UUID string")
