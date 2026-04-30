import json
import os
import gzip
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from services.graph_serialization import deserialize_graph, serialize_graph
from services.observability import timed_block, log_memory

GRAPH_KEY = "saved_graph.json"
WORKSPACE_STATE_KEY = "workspace_state.json"
LOCAL_GRAPH_FILE = "saved_graph.json"
LOCAL_WORKSPACE_FILE = "workspace_state.json"
STORAGE_COMPRESS_JSON = os.environ.get("STORAGE_COMPRESS_JSON", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def _s3_is_configured() -> bool:
    return all(
        os.environ.get(key)
        for key in ("S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    )


def _local_data_dir() -> Path:
    explicit_data_dir = os.environ.get("LOCAL_DATA_DIR", "").strip()
    if explicit_data_dir:
        data_dir = Path(explicit_data_dir).expanduser()
    elif os.environ.get("DESKTOP_APP_MODE", "").lower() == "true":
        data_dir = (
            Path.home()
            / "Library"
            / "Application Support"
            / "NeSyPaperGraph"
            / "data"
        )
    else:
        base_dir = Path(__file__).resolve().parents[1]
        data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _read_local_json(filename: str):
    path = _local_data_dir() / filename
    if not path.exists():
        return None
    with timed_block(f"local_json_read_{filename}"):
        with path.open("rb") as handle:
            blob = handle.read()
        return _decode_json_blob(blob)


def _write_local_json(filename: str, payload: dict) -> None:
    path = _local_data_dir() / filename
    with timed_block(f"local_json_write_{filename}"):
        with path.open("wb") as handle:
            handle.write(_encode_json_blob(payload))


def _encode_json_blob(payload: dict) -> bytes:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if STORAGE_COMPRESS_JSON:
        return gzip.compress(raw)
    return raw


def _decode_json_blob(blob: bytes, content_encoding: str = "") -> dict:
    if not blob:
        return {}
    should_try_gzip = STORAGE_COMPRESS_JSON or "gzip" in (content_encoding or "").lower()
    if should_try_gzip:
        try:
            blob = gzip.decompress(blob)
        except (OSError, EOFError):
            pass
    return json.loads(blob.decode("utf-8"))


def load_graph():
    """Download and deserialize the graph from S3 or local JSON."""
    if _s3_is_configured():
        bucket_name = os.environ["S3_BUCKET_NAME"]
        client = _s3_client()
        try:
            with timed_block("s3_get_object_load_graph"):
                response = client.get_object(Bucket=bucket_name, Key=GRAPH_KEY)
                payload = _decode_json_blob(
                    response["Body"].read(), response.get("ContentEncoding", "")
                )
                graph = deserialize_graph(payload)
            log_memory("s3_graph_loaded")
            return graph
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code not in {"NoSuchKey", "404"}:
                raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load graph from S3: {exc}") from exc

    payload = _read_local_json(LOCAL_GRAPH_FILE)
    if payload is None:
        return None
    return deserialize_graph(payload)


def save_graph(graph):
    """Serialize and upload the graph to S3 or local JSON."""
    payload = serialize_graph(graph)
    if _s3_is_configured():
        bucket_name = os.environ["S3_BUCKET_NAME"]
        client = _s3_client()
        with timed_block("s3_put_object_save_graph"):
            encoded_payload = _encode_json_blob(payload)
            put_kwargs = {
                "Bucket": bucket_name,
                "Key": GRAPH_KEY,
                "Body": encoded_payload,
                "ContentType": "application/json",
            }
            if STORAGE_COMPRESS_JSON:
                put_kwargs["ContentEncoding"] = "gzip"
            client.put_object(**put_kwargs)
        log_memory("s3_graph_saved")
        return
    _write_local_json(LOCAL_GRAPH_FILE, payload)
    log_memory("local_graph_saved")


def load_workspace_state():
    """Load workspace state from S3 or local JSON."""
    if _s3_is_configured():
        bucket_name = os.environ["S3_BUCKET_NAME"]
        client = _s3_client()
        try:
            with timed_block("s3_get_object_load_workspace"):
                response = client.get_object(Bucket=bucket_name, Key=WORKSPACE_STATE_KEY)
                return _decode_json_blob(
                    response["Body"].read(), response.get("ContentEncoding", "")
                )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code not in {"NoSuchKey", "404"}:
                raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load workspace state from S3: {exc}") from exc

    return _read_local_json(LOCAL_WORKSPACE_FILE)


def save_workspace_state(state):
    """Persist workspace state to S3 or local JSON."""
    if _s3_is_configured():
        bucket_name = os.environ["S3_BUCKET_NAME"]
        client = _s3_client()
        with timed_block("s3_put_object_save_workspace"):
            encoded_state = _encode_json_blob(state)
            put_kwargs = {
                "Bucket": bucket_name,
                "Key": WORKSPACE_STATE_KEY,
                "Body": encoded_state,
                "ContentType": "application/json",
            }
            if STORAGE_COMPRESS_JSON:
                put_kwargs["ContentEncoding"] = "gzip"
            client.put_object(**put_kwargs)
        return
    _write_local_json(LOCAL_WORKSPACE_FILE, state)
