import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
class FakeSparseVectorParams:
    modifier: str


@dataclass
class FakeSparseVector:
    indices: list
    values: list


@dataclass
class FakePointStruct:
    id: object
    vector: dict
    payload: dict


@dataclass
class FakeMatchValue:
    value: str


@dataclass
class FakeFieldCondition:
    key: str
    match: FakeMatchValue


@dataclass
class FakeFilter:
    must: list


@dataclass
class FakePrefetch:
    query: object
    using: str
    limit: int
    filter: object = None


@dataclass
class FakeFusionQuery:
    fusion: str


class FakeDistance:
    COSINE = "cosine"


class FakeModifier:
    IDF = "idf"


class FakeFusion:
    RRF = "rrf"


class FakeModels:
    Distance = FakeDistance
    FieldCondition = FakeFieldCondition
    Filter = FakeFilter
    Fusion = FakeFusion
    FusionQuery = FakeFusionQuery
    MatchValue = FakeMatchValue
    Modifier = FakeModifier
    PointStruct = FakePointStruct
    Prefetch = FakePrefetch
    SparseVector = FakeSparseVector
    SparseVectorParams = FakeSparseVectorParams
    VectorParams = FakeVectorParams


def patch_qdrant(
    monkeypatch,
    store,
    *,
    collection_exists=False,
    collection_info=None,
    query_points=None,
):
    clients = []
    default_collection_info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": object()},
                sparse_vectors={"bm25": object()},
            )
        )
    )

    class FakeQdrantClient:
        def __init__(self, url):
            self.url = url
            self.create_collection_calls = []
            self.delete_collection_calls = []
            self.get_collection_calls = []
            self.upsert_calls = []
            self.query_points_calls = []
            self.events = []
            clients.append(self)

        def collection_exists(self, collection_name):
            self.collection_exists_call = collection_name
            return collection_exists

        def get_collection(self, collection_name):
            self.get_collection_calls.append(collection_name)
            return collection_info or default_collection_info

        def create_collection(self, **kwargs):
            self.events.append(("create_collection", kwargs["collection_name"]))
            self.create_collection_calls.append(kwargs)

        def delete_collection(self, collection_name):
            self.events.append(("delete_collection", collection_name))
            self.delete_collection_calls.append(collection_name)

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
        "dense": [0.1, 0.2, 0.3],
        "sparse": {"indices": [5, 9], "values": [1.0, 2.0]},
        "payload": {
            "chunk_id": "chunk-hr-leave-0",
            "document_id": "doc-hr-leave",
            "source_path": "datasets/docs/hr/leave-policy.md",
            "title": "Annual leave policy",
        },
    }


def test_ensure_collection_creates_dense_and_sparse_vectors(monkeypatch):
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
            "vectors_config": {"dense": FakeVectorParams(size=384, distance=FakeDistance.COSINE)},
            "sparse_vectors_config": {"bm25": FakeSparseVectorParams(modifier=FakeModifier.IDF)},
        }
    ]


def test_ensure_collection_skips_create_when_existing_collection_has_hybrid_schema(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=True)

    store.ensure_collection("http://qdrant:6333", "chunks", 384)

    assert clients[0].collection_exists_call == "chunks"
    assert clients[0].get_collection_calls == ["chunks"]
    assert clients[0].create_collection_calls == []
    assert clients[0].delete_collection_calls == []


def test_ensure_collection_recreates_existing_dense_only_collection(monkeypatch):
    store = vector_store()
    dense_only_collection = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": object()},
                sparse_vectors={},
            )
        )
    )
    clients = patch_qdrant(
        monkeypatch,
        store,
        collection_exists=True,
        collection_info=dense_only_collection,
    )

    store.ensure_collection("http://qdrant:6333", "chunks", 384)

    client = clients[0]
    assert client.get_collection_calls == ["chunks"]
    assert client.delete_collection_calls == ["chunks"]
    assert client.create_collection_calls == [
        {
            "collection_name": "chunks",
            "vectors_config": {"dense": FakeVectorParams(size=384, distance=FakeDistance.COSINE)},
            "sparse_vectors_config": {"bm25": FakeSparseVectorParams(modifier=FakeModifier.IDF)},
        }
    ]
    assert client.events == [
        ("delete_collection", "chunks"),
        ("create_collection", "chunks"),
    ]


def test_ensure_collection_recreates_collection_missing_dense_vector(monkeypatch):
    store = vector_store()
    sparse_only_collection = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={},
                sparse_vectors={"bm25": object()},
            )
        )
    )
    clients = patch_qdrant(
        monkeypatch,
        store,
        collection_exists=True,
        collection_info=sparse_only_collection,
    )

    store.ensure_collection("http://qdrant:6333", "chunks", 384)

    client = clients[0]
    assert client.delete_collection_calls == ["chunks"]
    assert client.create_collection_calls[0]["collection_name"] == "chunks"


def test_delete_collection_if_exists_deletes_existing_collection(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=True)

    store.delete_collection_if_exists("http://qdrant:6333", "chunks")

    assert clients[0].collection_exists_call == "chunks"
    assert clients[0].delete_collection_calls == ["chunks"]


def test_delete_collection_if_exists_skips_missing_collection(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=False)

    store.delete_collection_if_exists("http://qdrant:6333", "chunks")

    assert clients[0].collection_exists_call == "chunks"
    assert clients[0].delete_collection_calls == []


@pytest.mark.parametrize("vector_size", [0, -1])
def test_ensure_collection_rejects_invalid_vector_size(vector_size):
    store = vector_store()

    with pytest.raises(ValueError, match="vector_size"):
        store.ensure_collection("http://qdrant:6333", "chunks", vector_size)


def test_upsert_chunk_vectors_passes_dense_and_sparse_named_vectors(monkeypatch):
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
                    vector={
                        "dense": [0.1, 0.2, 0.3],
                        "bm25": FakeSparseVector(indices=[5, 9], values=[1.0, 2.0]),
                    },
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
        ("dense", "dense"),
        ("sparse", "sparse"),
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


def test_upsert_chunk_vectors_rejects_empty_dense_vector(monkeypatch):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    point["dense"] = []

    with pytest.raises(ValueError, match="dense"):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


def test_upsert_chunk_vectors_rejects_mismatched_sparse_lengths(monkeypatch):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    point["sparse"] = {"indices": [1, 2], "values": [1.0]}

    with pytest.raises(ValueError, match="same length"):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


@pytest.mark.parametrize("point_id", ["point-1", -1, True, 1.5])
def test_upsert_chunk_vectors_rejects_qdrant_invalid_point_ids(monkeypatch, point_id):
    store = vector_store()
    patch_qdrant(monkeypatch, store)
    point = make_point()
    point["id"] = point_id

    with pytest.raises(ValueError, match="point id"):
        store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])


def test_upsert_chunk_vectors_accepts_unsigned_integer_point_id(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)
    point = make_point()
    point["id"] = 42

    store.upsert_chunk_vectors("http://qdrant:6333", "chunks", [point])

    assert clients[0].upsert_calls[0]["points"][0].id == 42


@pytest.mark.parametrize("payload_field", ["chunk_id", "document_id", "source_path", "title"])
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


def sparse_query():
    return {"indices": [5, 9], "values": [1.0, 2.0]}


def test_search_chunks_builds_hybrid_prefetch_with_rrf_fusion(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        sparse_query(),
        top_k=7,
    )

    call = clients[0].query_points_calls[0]
    assert call["limit"] == 7
    assert call["query"] == FakeFusionQuery(fusion=FakeFusion.RRF)
    assert call["with_payload"] is True
    assert call["with_vectors"] is False
    assert call["prefetch"] == [
        FakePrefetch(query=[0.1, 0.2], using="dense", limit=7, filter=None),
        FakePrefetch(
            query=FakeSparseVector(indices=[5, 9], values=[1.0, 2.0]),
            using="bm25",
            limit=7,
            filter=None,
        ),
    ]


def test_search_chunks_applies_metadata_filter_to_each_prefetch(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        sparse_query(),
        top_k=3,
        metadata_filter={"jang": "제2장 휴가", "jo": "제5조"},
    )

    expected_filter = FakeFilter(
        must=[
            FakeFieldCondition(key="jang", match=FakeMatchValue(value="제2장 휴가")),
            FakeFieldCondition(key="jo", match=FakeMatchValue(value="제5조")),
        ]
    )
    prefetch = clients[0].query_points_calls[0]["prefetch"]
    assert prefetch[0].filter == expected_filter
    assert prefetch[1].filter == expected_filter


def test_search_chunks_coerces_numeric_filter_values_to_int(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        sparse_query(),
        top_k=3,
        metadata_filter={"hang_no": "1", "jo": "제5조"},
    )

    # CLI 필터는 문자열로 오지만 hang_no/jo_no payload 는 int 이므로 정수로 맞춘다.
    must = clients[0].query_points_calls[0]["prefetch"][0].filter.must
    by_key = {condition.key: condition.match.value for condition in must}
    assert by_key["hang_no"] == 1
    assert by_key["jo"] == "제5조"


def test_search_chunks_without_sparse_terms_uses_dense_only(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        {"indices": [], "values": []},
        top_k=3,
    )

    prefetch = clients[0].query_points_calls[0]["prefetch"]
    assert len(prefetch) == 1
    assert prefetch[0].using == "dense"


def test_search_chunks_without_sparse_vector_uses_dense_only(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        None,
        top_k=3,
    )

    prefetch = clients[0].query_points_calls[0]["prefetch"]
    assert len(prefetch) == 1
    assert prefetch[0].using == "dense"


def test_search_chunks_rejects_sparse_query_missing_values(monkeypatch):
    store = vector_store()
    patch_qdrant(monkeypatch, store)

    with pytest.raises(ValueError, match="same length"):
        store.search_chunks(
            "http://qdrant:6333",
            "chunks",
            [0.1, 0.2],
            {"indices": [5, 9]},
            top_k=3,
        )


def test_search_chunks_rejects_sparse_query_mismatched_lengths(monkeypatch):
    store = vector_store()
    patch_qdrant(monkeypatch, store)

    with pytest.raises(ValueError, match="same length"):
        store.search_chunks(
            "http://qdrant:6333",
            "chunks",
            [0.1, 0.2],
            {"indices": [5, 9], "values": [1.0]},
            top_k=3,
        )


def test_search_chunks_returns_result_dicts(monkeypatch):
    store = vector_store()
    patch_qdrant(
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

    results = store.search_chunks("http://qdrant:6333", "chunks", [0.1, 0.2], sparse_query(), 1)

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
        store.search_chunks("http://qdrant:6333", "chunks", [0.1, 0.2], sparse_query(), top_k)


def test_search_chunks_rejects_empty_dense_vector():
    store = vector_store()

    with pytest.raises(ValueError, match="dense_vector"):
        store.search_chunks("http://qdrant:6333", "chunks", [], sparse_query(), 3)


def test_search_chunks_dense_mode_only_uses_dense_prefetch(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        sparse_query(),
        top_k=5,
        mode="dense",
    )

    prefetch = clients[0].query_points_calls[0]["prefetch"]
    assert len(prefetch) == 1
    assert prefetch[0].using == "dense"


def test_search_chunks_sparse_mode_only_uses_sparse_prefetch(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        sparse_query(),
        top_k=5,
        mode="sparse",
    )

    prefetch = clients[0].query_points_calls[0]["prefetch"]
    assert len(prefetch) == 1
    assert prefetch[0].using == "bm25"


def test_search_chunks_sparse_mode_without_terms_returns_empty_without_querying(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store)

    results = store.search_chunks(
        "http://qdrant:6333",
        "chunks",
        [0.1, 0.2],
        {"indices": [], "values": []},
        top_k=5,
        mode="sparse",
    )

    assert results == []
    assert clients == []


def test_search_chunks_rejects_invalid_mode():
    store = vector_store()

    with pytest.raises(ValueError, match="mode"):
        store.search_chunks(
            "http://qdrant:6333",
            "chunks",
            [0.1, 0.2],
            sparse_query(),
            top_k=3,
            mode="bogus",
        )
