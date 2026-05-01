import json
import os
import gzip
import logging
import time
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
LOCAL_BACKUP_RETENTION = int(
    os.environ.get("LOCAL_WORKSPACE_BACKUP_RETENTION", "20") or "20"
)
ALLOW_EMPTY_WORKSPACE_OVERWRITE = os.environ.get(
    "ALLOW_EMPTY_WORKSPACE_OVERWRITE", "false"
).lower() in {"1", "true", "yes", "y"}

logger = logging.getLogger(__name__)


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


def _local_backups_dir() -> Path:
    path = _local_data_dir() / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workspace_counts(payload: dict | None) -> tuple[int, int, int]:
    data = payload or {}
    reading_items = data.get("readingItems")
    theme_notes = data.get("themeNotes")
    annotations = data.get("paperAnnotations")
    return (
        len(reading_items) if isinstance(reading_items, list) else 0,
        len(theme_notes) if isinstance(theme_notes, list) else 0,
        len(annotations) if isinstance(annotations, dict) else 0,
    )


def _is_effectively_empty_workspace(payload: dict | None) -> bool:
    reading_count, theme_count, annotation_count = _workspace_counts(payload)
    return reading_count == 0 and theme_count == 0 and annotation_count == 0


def _iter_backup_files(filename: str) -> list[Path]:
    pattern = f"{filename}.*.bak"
    backups = sorted(
        _local_backups_dir().glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return backups


def _snapshot_existing_file(path: Path) -> None:
    if not path.exists():
        return
    timestamp = int(time.time() * 1000)
    backup_name = f"{path.name}.{timestamp}.bak"
    backup_path = _local_backups_dir() / backup_name
    backup_path.write_bytes(path.read_bytes())
    backups = _iter_backup_files(path.name)
    for stale in backups[LOCAL_BACKUP_RETENTION:]:
        try:
            stale.unlink()
        except OSError:
            logger.warning("Failed to delete stale backup: %s", stale)


def _read_local_json(filename: str):
    path = _local_data_dir() / filename
    if not path.exists():
        return None
    try:
        with timed_block(f"local_json_read_{filename}"):
            with path.open("rb") as handle:
                blob = handle.read()
            return _decode_json_blob(blob)
    except Exception as exc:
        logger.warning(
            "Failed to decode local JSON file %s; treating as missing state: %s",
            path,
            exc,
        )
        for backup_path in _iter_backup_files(filename):
            try:
                with backup_path.open("rb") as handle:
                    blob = handle.read()
                restored = _decode_json_blob(blob)
                logger.warning(
                    "Recovered local JSON file %s from backup %s",
                    path,
                    backup_path,
                )
                return restored
            except Exception:
                continue
        return None


def _write_local_json(filename: str, payload: dict) -> None:
    path = _local_data_dir() / filename
    with timed_block(f"local_json_write_{filename}"):
        _snapshot_existing_file(path)
        encoded = _encode_json_blob(payload)
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)


def _encode_json_blob(payload: dict) -> bytes:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if STORAGE_COMPRESS_JSON:
        return gzip.compress(raw)
    return raw


def _decode_json_blob(blob: bytes, content_encoding: str = "") -> dict:
    if not blob:
        return {}
    gzip_magic = len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B
    should_try_gzip = (
        gzip_magic
        or STORAGE_COMPRESS_JSON
        or "gzip" in (content_encoding or "").lower()
    )
    errors = []
    decode_modes = []
    if should_try_gzip:
        decode_modes.append("gzip")
    decode_modes.append("plain")
    if not should_try_gzip:
        decode_modes.append("gzip")

    for mode in decode_modes:
        try:
            candidate = gzip.decompress(blob) if mode == "gzip" else blob
            return json.loads(candidate.decode("utf-8"))
        except Exception as exc:
            errors.append(f"{mode}:{exc}")
            continue
    raise ValueError(
        "Unable to decode JSON blob. Errors: " + " | ".join(errors[:4])
    )


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
    if not ALLOW_EMPTY_WORKSPACE_OVERWRITE:
        existing_state = load_workspace_state()
        if (
            existing_state is not None
            and not _is_effectively_empty_workspace(existing_state)
            and _is_effectively_empty_workspace(state)
        ):
            existing_counts = _workspace_counts(existing_state)
            raise RuntimeError(
                "Refusing to overwrite non-empty workspace with empty state payload. "
                f"Existing counts={existing_counts}. Set ALLOW_EMPTY_WORKSPACE_OVERWRITE=true "
                "to bypass."
            )
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
