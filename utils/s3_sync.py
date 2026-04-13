"""S3 backup utilities for syncing data, models, and evaluation artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class S3Syncer:
    """Upload/download files and directories to/from S3.

    Parameters
    ----------
    bucket : str
        S3 bucket name (e.g. ``"rnd-sagemaker"``).
    prefix : str
        Key prefix within the bucket (e.g. ``"Multimodal_CSI_Text"``).
    """

    SUBFOLDER_MODEL = "Model"
    SUBFOLDER_EVALUATION = "Evaluation"
    SUBFOLDER_DATASET = "Dataset"

    def __init__(self, bucket: str, prefix: str = "Multimodal_CSI_Text") -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        try:
            import boto3
            self._s3 = boto3.client("s3")
            logger.info("S3Syncer ready: s3://%s/%s/", bucket, prefix)
        except ImportError:
            raise ImportError("boto3 is required for S3 sync. pip install boto3")

    def _s3_key(self, subfolder: str, relative_path: str) -> str:
        """Build the full S3 key."""
        return f"{self.prefix}/{subfolder}/{relative_path}"

    def upload_file(self, local_path: str, subfolder: str, key_name: Optional[str] = None) -> str:
        """Upload a single file to S3.

        Returns the full S3 URI of the uploaded object.
        """
        local = Path(local_path)
        if not local.is_file():
            raise FileNotFoundError(f"Local file not found: {local}")

        if key_name is None:
            key_name = local.name

        s3_key = self._s3_key(subfolder, key_name)
        self._s3.upload_file(str(local), self.bucket, s3_key)
        uri = f"s3://{self.bucket}/{s3_key}"
        logger.info("Uploaded %s -> %s", local, uri)
        return uri

    def upload_directory(self, local_dir: str, subfolder: str, prefix: str = "") -> int:
        """Recursively upload a directory to S3.

        Returns the number of files uploaded.
        """
        local = Path(local_dir)
        if not local.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local}")

        count = 0
        for fpath in local.rglob("*"):
            if fpath.is_file():
                relative = fpath.relative_to(local)
                key_name = f"{prefix}/{relative}" if prefix else str(relative)
                s3_key = self._s3_key(subfolder, key_name)
                self._s3.upload_file(str(fpath), self.bucket, s3_key)
                count += 1

        logger.info(
            "Uploaded %d files from %s -> s3://%s/%s/%s/",
            count, local, self.bucket, self.prefix, subfolder,
        )
        return count

    def download_file(self, subfolder: str, key_name: str, local_path: str) -> Path:
        """Download a single file from S3."""
        s3_key = self._s3_key(subfolder, key_name)
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, s3_key, str(local))
        logger.info("Downloaded s3://%s/%s -> %s", self.bucket, s3_key, local)
        return local

    def download_directory(self, subfolder: str, local_dir: str, prefix: str = "") -> int:
        """Download all objects under a prefix to a local directory.

        Returns the number of files downloaded.
        """
        s3_prefix = self._s3_key(subfolder, prefix)
        local = Path(local_dir)

        paginator = self._s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative = s3_key[len(s3_prefix):].lstrip("/")
                if not relative:
                    continue
                dest = local / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self.bucket, s3_key, str(dest))
                count += 1

        logger.info(
            "Downloaded %d files from s3://%s/%s -> %s",
            count, self.bucket, s3_prefix, local,
        )
        return count

    # Convenience methods for the three artifact types

    def upload_checkpoint(self, local_path: str, key_name: Optional[str] = None) -> str:
        """Upload a model checkpoint."""
        return self.upload_file(local_path, self.SUBFOLDER_MODEL, key_name)

    def upload_evaluation(self, local_path: str, key_name: Optional[str] = None) -> str:
        """Upload an evaluation result file."""
        if Path(local_path).is_dir():
            self.upload_directory(local_path, self.SUBFOLDER_EVALUATION)
            return f"s3://{self.bucket}/{self._s3_key(self.SUBFOLDER_EVALUATION, '')}"
        return self.upload_file(local_path, self.SUBFOLDER_EVALUATION, key_name)

    def upload_dataset(self, local_path: str, key_name: Optional[str] = None) -> str:
        """Upload a dataset artifact (pair registry, text embeddings, etc.)."""
        if Path(local_path).is_dir():
            self.upload_directory(local_path, self.SUBFOLDER_DATASET)
            return f"s3://{self.bucket}/{self._s3_key(self.SUBFOLDER_DATASET, '')}"
        return self.upload_file(local_path, self.SUBFOLDER_DATASET, key_name)

    def sync_to_s3(self, local_path: str, subfolder: str) -> None:
        """Generic sync: upload file or directory."""
        p = Path(local_path)
        if p.is_file():
            self.upload_file(local_path, subfolder)
        elif p.is_dir():
            self.upload_directory(local_path, subfolder)
        else:
            logger.warning("sync_to_s3: path does not exist: %s", local_path)

    def sync_from_s3(self, subfolder: str, local_path: str, prefix: str = "") -> None:
        """Generic sync: download from S3 prefix to local directory."""
        self.download_directory(subfolder, local_path, prefix)

    def __repr__(self) -> str:
        return f"S3Syncer(bucket={self.bucket!r}, prefix={self.prefix!r})"
