"""
Unit tests for the S3Uploader module.
Verifies Hive partitioning key formatting, in-memory gzip compression,
bucket self-healing, and graceful handling of network/AWS errors.
"""

import gzip
import io
import json
from unittest.mock import MagicMock, patch
import pytest
from botocore.exceptions import ClientError, BotoCoreError

from src.sync.s3_uploader import S3Uploader


@pytest.fixture
def mock_boto_client():
    """Mocks the boto3 S3 client to isolate tests from actual network calls."""
    with patch("boto3.client") as mock_client_factory:
        mock_s3 = MagicMock()
        mock_client_factory.return_value = mock_s3
        yield mock_s3


def test_s3_uploader_initialization_and_bucket_creation(mock_boto_client):
    """Verifies that S3Uploader creates/verifies the bucket upon initialization."""
    uploader = S3Uploader(
        endpoint_url="http://localhost:4566",
        bucket_name="test-bucket",
        device_id="edge-dev-01"
    )

    assert uploader.bucket_name == "test-bucket"
    assert uploader.device_id == "edge-dev-01"
    mock_boto_client.create_bucket.assert_called_once_with(Bucket="test-bucket")


def test_s3_uploader_successful_upload_with_hive_partitioning(mock_boto_client):
    """
    Validates successful upload to S3.
    Ensures Hive partitioning path format: year=YYYY/month=MM/day=DD/device_id=ID/
    and confirms in-memory Gzip payload compression.
    """
    uploader = S3Uploader(
        endpoint_url="http://localhost:4566",
        bucket_name="vibration-bucket",
        device_id="sensor-node-01"
    )

    # Fixed timestamp: July 27, 2026 UTC
    test_timestamp = 1785110400.0
    payload = {
        "timestamp": test_timestamp,
        "features": {"rms": 1.5, "std": 0.4},
        "anomaly_detected": False
    }
    batch_id = "batch_12345.json.gz"

    success = uploader.upload(batch_id=batch_id, payload=payload)

    assert success is True

    # Inspect the call made to put_object
    mock_boto_client.put_object.assert_called_once()
    call_kwargs = mock_boto_client.put_object.call_args[1]

    # Verify Bucket and Hive Partitioning S3 Key Structure
    assert call_kwargs["Bucket"] == "vibration-bucket"
    assert call_kwargs["Key"] == "year=2026/month=07/day=27/device_id=sensor-node-01/batch_12345.json.gz"
    assert call_kwargs["ContentEncoding"] == "gzip"
    assert call_kwargs["ContentType"] == "application/json"

    # Decompress and verify body content integrity
    compressed_bytes = call_kwargs["Body"]
    with gzip.GzipFile(fileobj=io.BytesIO(compressed_bytes), mode="rb") as gz:
        decompressed_data = json.loads(gz.read().decode("utf-8"))

    assert decompressed_data["timestamp"] == test_timestamp
    assert decompressed_data["features"]["rms"] == 1.5


def test_s3_uploader_self_healing_on_missing_bucket(mock_boto_client):
    """
    Simulates a NoSuchBucket error during put_object.
    Ensures S3Uploader attempts self-healing (re-creates bucket) and retries the upload.
    """
    uploader = S3Uploader(bucket_name="auto-heal-bucket")

    # Reset mock after init call
    mock_boto_client.reset_mock()

    # Raise NoSuchBucket ClientError on the first put_object call
    no_bucket_error = ClientError(
        error_response={"Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"}},
        operation_name="PutObject"
    )
    
    # First call raises error, second call succeeds
    mock_boto_client.put_object.side_effect = [no_bucket_error, {"ResponseMetadata": {"HTTPStatusCode": 200}}]

    payload = {"timestamp": 1785110400.0, "status": "ok"}
    success = uploader.upload(batch_id="batch_retry.json.gz", payload=payload)

    assert success is True
    # Should have attempted bucket creation again during self-healing
    mock_boto_client.create_bucket.assert_called_with(Bucket="auto-heal-bucket")
    # Should have called put_object twice (initial fail + retry success)
    assert mock_boto_client.put_object.call_count == 2


def test_s3_uploader_returns_false_on_network_failure(mock_boto_client):
    """
    Validates that connection drops or AWS exceptions return False,
    allowing the edge pipeline to retain the payload safely in the DurableQueue.
    """
    uploader = S3Uploader(bucket_name="failing-bucket")

    # Simulate a total connection error (e.g., LocalStack container stopped/down)
    mock_boto_client.put_object.side_effect = BotoCoreError()

    payload = {"timestamp": 1785110400.0, "data": "test"}
    success = uploader.upload(batch_id="batch_fail.json.gz", payload=payload)

    assert success is False