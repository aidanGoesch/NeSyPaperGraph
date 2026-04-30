import io

from models.graph import PaperGraph


def test_encode_decode_json_blob_roundtrip(monkeypatch):
    import services.storage_service as storage_service

    payload = {"a": 1, "nested": {"k": "v"}, "items": [1, 2, 3]}
    monkeypatch.setattr(storage_service, "STORAGE_COMPRESS_JSON", True)
    encoded = storage_service._encode_json_blob(payload)
    decoded = storage_service._decode_json_blob(encoded, "gzip")
    assert decoded == payload


def test_local_workspace_state_roundtrip(tmp_path, monkeypatch):
    import services.storage_service as storage_service

    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path / "data"))
    data = {"readingItems": [], "themeNotes": [], "paperAnnotations": {}}
    storage_service.save_workspace_state(data)
    loaded = storage_service.load_workspace_state()
    assert loaded == data


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
