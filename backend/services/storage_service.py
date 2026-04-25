import io
import os
import pickle

import boto3
from botocore.exceptions import ClientError

GRAPH_KEY = "saved_graph.pkl"


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def load_graph():
    """Download and deserialize the graph from S3. Returns None if not found."""
    bucket_name = os.environ["S3_BUCKET_NAME"]
    client = _s3_client()
    try:
        response = client.get_object(Bucket=bucket_name, Key=GRAPH_KEY)
        return pickle.loads(response["Body"].read())
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "NoSuchBucket", "404"}:
            return None
        raise


def save_graph(graph):
    """Serialize and upload the graph to S3."""
    bucket_name = os.environ["S3_BUCKET_NAME"]
    client = _s3_client()
    buffer = io.BytesIO()
    pickle.dump(graph, buffer)
    buffer.seek(0)
    client.put_object(Bucket=bucket_name, Key=GRAPH_KEY, Body=buffer.getvalue())
