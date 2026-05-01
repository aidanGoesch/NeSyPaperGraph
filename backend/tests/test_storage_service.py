import io
import gzip
import json
from pathlib import Path

from models.graph import PaperGraph


def test_encode_decode_json_blob_roundtrip(monkeypatch):
    import services.storage_service as storage_service

    payload = {"a": 1, "nested": {"k": "v"}, "items": [1, 2, 3]}
    monkeypatch.setattr(storage_service, "STORAGE_COMPRESS_JSON", True)
    encoded = storage_service._encode_json_blob(payload)
    decoded = storage_service._decode_json_blob(encoded, "gzip")
    assert decoded == payload


def test_decode_json_blob_detects_gzip_magic_when_flag_disabled(monkeypatch):
    import services.storage_service as storage_service

    payload = {"readingItems": [], "themeNotes": [], "paperAnnotations": {}}
    monkeypatch.setattr(storage_service, "STORAGE_COMPRESS_JSON", False)
    encoded = gzip.compress(json.dumps(payload).encode("utf-8"))
    decoded = storage_service._decode_json_blob(encoded, "")
    assert decoded == payload


def test_local_workspace_state_roundtrip(tmp_path, monkeypatch):
    import services.storage_service as storage_service

    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "data"))
    data = {"readingItems": [], "themeNotes": [], "paperAnnotations": {}}
    storage_service.save_workspace_state(data)
    loaded = storage_service.load_workspace_state()
    assert loaded == data


def test_read_local_json_recovers_from_backup_when_primary_corrupt(tmp_path, monkeypatch):
    import services.storage_service as storage_service

    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "data"))
    first = {"readingItems": [{"id": "a"}], "themeNotes": [], "paperAnnotations": {}}
    second = {"readingItems": [{"id": "b"}], "themeNotes": [], "paperAnnotations": {}}
    storage_service._write_local_json(storage_service.LOCAL_WORKSPACE_FILE, first)
    storage_service._write_local_json(storage_service.LOCAL_WORKSPACE_FILE, second)
    workspace_path = (
        Path(storage_service._local_data_dir()) / storage_service.LOCAL_WORKSPACE_FILE
    )
    workspace_path.write_bytes(b"corrupt-not-json")

    recovered = storage_service._read_local_json(storage_service.LOCAL_WORKSPACE_FILE)

    assert recovered == first


def test_save_workspace_state_rejects_empty_overwrite(monkeypatch):
    import services.storage_service as storage_service

    monkeypatch.setattr(storage_service, "ALLOW_EMPTY_WORKSPACE_OVERWRITE", False)
    monkeypatch.setattr(
        storage_service,
        "load_workspace_state",
        lambda: {
            "readingItems": [{"id": "read-1"}],
            "themeNotes": [{"id": "theme-1"}],
            "paperAnnotations": {},
        },
    )

    try:
        storage_service.save_workspace_state(
            {"readingItems": [], "themeNotes": [], "paperAnnotations": {}}
        )
        assert False, "Expected RuntimeError for empty overwrite guard"
    except RuntimeError as exc:
        assert "Refusing to overwrite non-empty workspace" in str(exc)


def test_save_and_load_graph_local(tmp_path, monkeypatch):
    import services.storage_service as storage_service
    from services.graph_serialization import serialize_graph

    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "data"))
    graph = PaperGraph()
    graph.add_paper(
        __import__("models.paper", fromlist=["Paper"]).Paper(
            title="Paper A", file_path="a.pdf", topics=["Topic A"], summary="s"
        )
    )
    storage_service.save_graph(graph)
    loaded = storage_service.load_graph()
    assert loaded is not None
    assert "Paper A" in loaded.graph.nodes
    assert serialize_graph(loaded)["version"] == 1


def test_load_graph_s3_no_such_key(monkeypatch):
    import services.storage_service as storage_service

    class FakeClientError(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "NoSuchKey"}}

    class FakeClient:
        def get_object(self, **_kwargs):
            raise storage_service.ClientError(
                {"Error": {"Code": "NoSuchKey"}},
                "GetObject",
            )

    monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(storage_service, "_s3_client", lambda: FakeClient())
    assert storage_service.load_graph() is None


def test_save_graph_s3_uses_gzip(monkeypatch):
    import services.storage_service as storage_service

    captured = {}

    class FakeClient:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(storage_service, "_s3_client", lambda: FakeClient())
    monkeypatch.setattr(storage_service, "STORAGE_COMPRESS_JSON", True)
    monkeypatch.setattr(storage_service, "serialize_graph", lambda _g: {"ok": True})

    storage_service.save_graph(PaperGraph())
    assert captured["Bucket"] == "bucket"
    assert captured["Key"] == storage_service.GRAPH_KEY
    assert captured["ContentEncoding"] == "gzip"
    assert isinstance(captured["Body"], bytes)
