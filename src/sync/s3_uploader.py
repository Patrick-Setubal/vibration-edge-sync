"""
S3 Uploader module.
Handles data transmission to AWS S3 or LocalStack.
Enforces partition key hierarchies (year/month/day/device_id) for efficient querying.
Includes self-healing mechanisms for missing buckets/initialization delays.
"""

import json
import gzip
import io
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config import S3_ENDPOINT_URL, S3_BUCKET_NAME, DEVICE_ID
from src.utils.logger import get_logger

logger = get_logger(__name__)


class S3Uploader:
    """
    Client interface for uploading processed edge batches to AWS S3 (or LocalStack mock).
    Formats S3 object paths with Hive-style time partitioning to reduce scan costs on Athena/S3 Select.
    Self-heals missing buckets dynamically upon upload failures.
    """

    def __init__(
        self,
        endpoint_url: str = S3_ENDPOINT_URL,
        bucket_name: str = S3_BUCKET_NAME,
        device_id: str = DEVICE_ID
    ):
        """
        Args:
            endpoint_url: Endpoint URL for LocalStack or custom S3 compatible storage.
            bucket_name: Target S3 bucket name.
            device_id: Identifier of the edge device producing data.
        """
        self.bucket_name = bucket_name
        self.device_id = device_id
        
        # Initialize boto3 S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url if endpoint_url else None,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test"
        )
        
        # Best-effort bucket creation on init (won't crash if LocalStack is still starting up)
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> bool:
        """
        Helper method to automatically verify and create the bucket if missing.
        Returns True if bucket exists/was created, False on error.
        """
        try:
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 Bucket '{self.bucket_name}' verified/created.")
            return True
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"Could not verify/create S3 bucket '{self.bucket_name}': {e}")
            return False

    def upload(self, batch_id: str, payload: Dict[str, Any]) -> bool:
        """
        Uploads a compressed payload to S3 with dynamic Hive partitioning.
        Self-heals by retrying bucket creation if a NoSuchBucket error occurs.
        
        Args:
            batch_id: Unique name of the batch file.
            payload: Payload dictionary containing window features and timestamps.
            
        Returns:
            bool: True if upload succeeded, False if a network or server error occurred.
        """
        # Determine partitioning path based on the batch timestamp
        timestamp = payload.get("timestamp", datetime.now(timezone.utc).timestamp())
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        s3_key = (
            f"year={dt.year:04d}/"
            f"month={dt.month:02d}/"
            f"day={dt.day:02d}/"
            f"device_id={self.device_id}/"
            f"{batch_id}"
        )

        # Re-compress payload in memory for streaming upload
        json_data = json.dumps(payload).encode("utf-8")
        compressed_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz:
            gz.write(json_data)
        
        compressed_bytes = compressed_buffer.getvalue()

        # Attempt PutObject (with retry logic for missing buckets)
        try:
            return self._put_object_to_s3(s3_key, compressed_bytes)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # If the bucket doesn't exist, try creating it on the fly and re-uploading
            if error_code in ("NoSuchBucket", "404"):
                logger.warning(f"Bucket '{self.bucket_name}' missing during upload. Attempting self-healing...")
                if self._ensure_bucket_exists():
                    try:
                        return self._put_object_to_s3(s3_key, compressed_bytes)
                    except (BotoCoreError, ClientError) as retry_err:
                        logger.error(f"Re-upload failed after bucket creation: {retry_err}")
                        return False
            
            logger.error(f"S3 ClientError uploading {batch_id}: {e}")
            return False
        except (BotoCoreError, Exception) as e:
            logger.error(f"Unexpected error uploading {batch_id}: {e}", exc_info=True)
            return False

    def _put_object_to_s3(self, s3_key: str, data_bytes: bytes) -> bool:
        """Internal helper to execute the put_object SDK call."""
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=data_bytes,
            ContentEncoding="gzip",
            ContentType="application/json"
        )
        logger.info(f"Successfully uploaded batch to s3://{self.bucket_name}/{s3_key}")
        return True