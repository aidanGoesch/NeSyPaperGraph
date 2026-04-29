import json
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from services.graph_serialization import deserialize_graph, serialize_graph
from services.observability import timed_block, log_memory

GRAPH_KEY = "saved_graph.json"
WORKSPACE_STATE_KEY = "workspace_state.json"
LOCAL_GRAPH_FILE = "saved_graph.json"
LOCAL_WORKSPACE_FILE = "workspace_state.json"


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
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("LOCAL_DATA_DIR", str(base_dir / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _read_local_json(filename: str):
    path = _local_data_dir() / filename
    if not path.exists():
        return None
    with timed_block(f"local_json_read_{filename}"):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def _write_local_json(filename: str, payload: dict) -> None:
    path = _local_data_dir() / filename
    with timed_block(f"local_json_write_{filename}"):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def load_graph():
    """Download and deserialize the graph from S3 or local JSON."""
    if _s3_is_configured():
        bucket_name = os.environ["S3_BUCKET_NAME"]
        client = _s3_client()
        try:
            with timed_block("s3_get_object_load_graph"):
                response = client.get_object(Bucket=bucket_name, Key=GRAPH_KEY)
                payload = json.loads(response["Body"].read().decode("utf-8"))
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
            client.put_object(
                Bucket=bucket_name,
                Key=GRAPH_KEY,
                Body=json.dumps(payload).encode("utf-8"),
                ContentType="application/json",
            )
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
                return json.loads(response["Body"].read().decode("utf-8"))
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
            client.put_object(
                Bucket=bucket_name,
                Key=WORKSPACE_STATE_KEY,
                Body=json.dumps(state).encode("utf-8"),
                ContentType="application/json",
            )
        return
    _write_local_json(LOCAL_WORKSPACE_FILE, state)
