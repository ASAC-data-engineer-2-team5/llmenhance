from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def vector_store():
    try:
        return importlib.import_module("app.vector_store")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.vector_store should exist: {exc}")


@dataclass
class FakeVectorParams:
    size: int
    distance: str


@dataclass
class FakePointStruct:
    id: object
    vector: list[float]
    payload: dict


@dataclass
class FakeMatchAny:
    any: list[str]


@dataclass
class FakeFieldCondition:
    key: str
    match: FakeMatchAny


@dataclass
class FakeFilter:
    must: list[FakeFieldCondition]


class FakeDistance:
    COSINE = "cosine"


class FakeModels:
    Distance = FakeDistance
    FieldCondition = FakeFieldCondition
    Filter = FakeFilter
    MatchAny = FakeMatchAny
    PointStruct = FakePointStruct
    VectorParams = FakeVectorParams


def patch_qdrant(monkeypatch, store, *, collection_exists=False, query_points=None):
    clients = []

    class FakeQdrantClient:
        def __init__(self, url):
            self.url = url
            self.create_collection_calls = []
            self.upsert_calls = []
            self.query_points_calls = []
            clients.append(self)

        def collection_exists(self, collection_name):
            self.collection_exists_call = collection_name
            return collection_exists

        def create_collection(self, **kwargs):
            self.create_collection_calls.append(kwargs)

        def upsert(self, **kwargs):
            self.upsert_calls.append(kwargs)

        def query_points(self, **kwargs):
            self.query_points_calls.append(kwargs)
            return query_points or SimpleNamespace(points=[])

    monkeypatch.setattr(store, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(store, "models", FakeModels)
    return clients


def make_point():
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "vector": [0.1, 0.2, 0.3],
        "payload": {
            "chunk_id": "chunk-hr-leave-0",
            "document_id": "doc-hr-leave",
            "source_path": "datasets/docs/hr/leave-policy.md",
            "title": "Annual leave policy",
        },
    }


def test_ensure_collection_creates_collection_with_cosine_distance(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=False)

    store.ensure_collection(
        qdrant_url="http://qdrant:6333",
        collection_name="chunks",
        vector_size=384,
    )

    client = clients[0]
    assert client.url == "http://qdrant:6333"
    assert client.collection_exists_call == "chunks"
    assert client.create_collection_calls == [
        {
            "collection_name": "chunks",
            "vectors_config": FakeVectorParams(size=384, distance=FakeDistance.COSINE),
        }
    ]


def test_ensure_collection_skips_create_when_collection_exists(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=True)

    store.ensure_collection("http://qdrant:6333", "chunks", 384)

    assert clients[0].collection_exists_call == "chunks"
    assert clients[0].create_collection_calls == []


@pytest.mark.parametrize("vector_size", [0, -1])
def test_ensure_collection_rejects_invalid_vector_size(vector_size):
    store = vector_store()

    with pytest.raises(ValueError, match="vector_size"):
        store.ensure_collection("http://qdrant:6333", "chunks", vector_size)


def test_upsert_chunk_vectors_passes_vectors_and_payload_to_qdrant(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)
    point = make_point()

    store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])

    assert clients[0].upsert_calls == [
        {
            "collection_name": "chunks",
            "points": [
                FakePointStruct(
                    id="550e8400-e29b-41d4-a716-446655440000",
                    vector=[0.1, 0.2, 0.3],
                    payload=point["payload"],
                )
            ],
        }
    ]


def test_upsert_chunk_vectors_empty_points_does_not_call_upsert(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [])

    assert clients == []


@pytest.mark.parametrize(
    ("field_to_remove", "expected_message"),
    [
        ("id", "id"),
        ("vector", "vector"),
        ("payload.chunk_id", "chunk_id"),
        ("payload.document_id", "document_id"),
        ("payload.source_path", "source_path"),
        ("payload.title", "title"),
    ],
)
def test_upsert_chunk_vectors_rejects_missing_required_fields(
    monkeypatch,
    field_to_remove,
    expected_message,
):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    if field_to_remove.startswith("payload."):
        _, payload_field = field_to_remove.split(".", maxsplit=1)
        del point["payload"][payload_field]
    else:
        del point[field_to_remove]

    with pytest.raises(ValueError, match=expected_message):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


@pytest.mark.parametrize("point_id", ["point-1", -1, True, 1.5])
def test_upsert_chunk_vectors_rejects_qdrant_invalid_point_ids(
    monkeypatch,
    point_id,
):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    point["id"] = point_id

    with pytest.raises(ValueError, match="point id"):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


def test_upsert_chunk_vectors_rejects_integer_point_id_above_uint64(monkeypatch):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    point["id"] = 2**64

    with pytest.raises(ValueError, match="point id"):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


def test_upsert_chunk_vectors_accepts_max_uint64_point_id(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)
    point = make_point()
    point["id"] = 2**64 - 1

    store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])

    assert clients[0].upsert_calls[0]["points"][0].id == 2**64 - 1


def test_upsert_chunk_vectors_accepts_unsigned_integer_point_id(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)
    point = make_point()
    point["id"] = 42

    store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])

    assert clients[0].upsert_calls[0]["points"][0].id == 42


@pytest.mark.parametrize(
    "payload_field",
    ["chunk_id", "document_id", "source_path", "title"],
)
@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_upsert_chunk_vectors_rejects_empty_or_non_string_payload_values(
    monkeypatch,
    payload_field,
    bad_value,
):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    point["payload"][payload_field] = bad_value

    with pytest.raises(ValueError, match=payload_field):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


def test_search_chunks_passes_top_k_as_limit(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks("http://qdrant:6333", "chunks", [0.1, 0.2], top_k=7)

    assert clients[0].query_points_calls[0]["limit"] == 7


def test_search_chunks_without_candidates_uses_no_filter(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        top_k=3,
        candidate_chunk_ids=None,
    )

    assert clients[0].query_points_calls[0]["query_filter"] is None


def test_search_chunks_with_candidates_filters_by_chunk_id(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        top_k=3,
        candidate_chunk_ids=["chunk-1", "chunk-2"],
    )

    query_filter = clients[0].query_points_calls[0]["query_filter"]
    assert query_filter == FakeFilter(
        must=[
            FakeFieldCondition(
                key="chunk_id",
                match=FakeMatchAny(any=["chunk-1", "chunk-2"]),
            )
        ]
    )


def test_search_chunks_empty_candidates_returns_empty_without_querying(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    results = store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        top_k=3,
        candidate_chunk_ids=[],
    )

    assert results == []
    assert clients == []


def test_search_chunks_returns_result_dicts(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(
        monkeypatch,
        store,
        query_points=SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.92,
                    payload={"chunk_id": "chunk-hr-leave-0"},
                )
            ]
        ),
    )

    results = store.search_chunks("http://qdrant:6333", "chunks", [0.1, 0.2], 1)

    assert clients[0].query_points_calls[0] == {
        "collection_name": "chunks",
        "query": [0.1, 0.2],
        "limit": 1,
        "query_filter": None,
        "with_payload": True,
        "with_vectors": False,
    }
    assert results == [
        {
            "id": "point-1",
            "score": 0.92,
            "payload": {"chunk_id": "chunk-hr-leave-0"},
        }
    ]


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_chunks_rejects_invalid_top_k(top_k):
    store = vector_store()

    with pytest.raises(ValueError, match="top_k"):
        store.search_chunks("http://qdrant:6333", "chunks", [0.1, 0.2], top_k)


def test_search_chunks_rejects_empty_query_vector():
    store = vector_store()

    with pytest.raises(ValueError, match="query_vector"):
        store.search_chunks("http://qdrant:6333", "chunks", [], 3)
