"""
Generic File Uploader Utility

Supports:
- Upload to S3
- Upload to local filesystem
- Upload from file path or bytes
- Async + sync APIs

Usage:
    from pycorekit.utils.uploader import upload_file, upload_bytes

    # Upload a local file to S3
    
    from pycorekit.utils.uploader import upload_file

    await upload_file(
        "temp/report.pdf",
        dest="s3",
        bucket="my-bucket",
        key_prefix="documents",
    )

    # Upload raw bytes to local storage

    await upload_file(
    "temp/report.pdf",
    dest="local",
    dest="uploads",
) 
"""

import os
import boto3
import aiofiles
from typing import Optional


# ---------------------------------------------------------
# LOCAL UPLOAD
# ---------------------------------------------------------

async def _upload_local(
    file_path: str,
    dest_dir: str,
    dest_name: Optional[str] = None,
) -> str:
    """
    Copy a file to a local directory.
    """
    os.makedirs(dest_dir, exist_ok=True)

    dest_name = dest_name or os.path.basename(file_path)
    dest_path = os.path.join(dest_dir, dest_name)

    async with aiofiles.open(file_path, "rb") as src:
        async with aiofiles.open(dest_path, "wb") as dst:
            chunk = await src.read(1024 * 64)
            while chunk:
                await dst.write(chunk)
                chunk = await src.read(1024 * 64)

    return dest_path


async def _upload_local_bytes(
    data: bytes,
    dest_dir: str,
    dest_name: str,
) -> str:
    """
    Save raw bytes to a local directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, dest_name)

    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(data)

    return dest_path


# ---------------------------------------------------------
# S3 UPLOAD
# ---------------------------------------------------------

def _upload_s3(
    file_path: str,
    bucket: str,
    key_prefix: str = "",
    region: str = "us-east-1",
    dest_name: Optional[str] = None,
) -> str:
    """
    Upload a file to S3 synchronously.
    """
    s3 = boto3.client("s3", region_name=region)

    dest_name = dest_name or os.path.basename(file_path)
    key = f"{key_prefix}/{dest_name}" if key_prefix else dest_name

    s3.upload_file(file_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _upload_s3_bytes(
    data: bytes,
    bucket: str,
    key: str,
    region: str = "us-east-1",
) -> str:
    """
    Upload raw bytes to S3.
    """
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

async def upload_file(
    file_path: str,
    dest: str,
    *,
    bucket: Optional[str] = None,
    key_prefix: str = "",
    region: str = "us-east-1",
    dest_name: Optional[str] = None,
) -> str:
    """
    Upload a file to:
    - Local directory
    - S3 bucket

    Args:
        file_path: Local file path to upload
        dest: Local directory OR "s3"
        bucket: Required if dest="s3"
        key_prefix: Optional S3 prefix
        region: AWS region
        dest_name: Optional override for destination filename

    Returns:
        str: Destination path or S3 URI
    """

    if dest == "s3":
        if not bucket:
            raise ValueError("bucket is required when dest='s3'")
        return _upload_s3(file_path, bucket, key_prefix, region, dest_name)

    # Local upload
    return await _upload_local(file_path, dest, dest_name)


async def upload_bytes(
    data: bytes,
    dest: str,
    *,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    region: str = "us-east-1",
    dest_dir: Optional[str] = None,
    dest_name: Optional[str] = None,
) -> str:
    """
    Upload raw bytes to:
    - Local directory
    - S3 bucket

    Args:
        data: Raw bytes
        dest: "s3" or "local"
        bucket: Required if dest="s3"
        key: Required if dest="s3"
        dest_dir: Required if dest="local"
        dest_name: Required if dest="local"

    Returns:
        str: Destination path or S3 URI
    """

    if dest == "s3":
        if not bucket or not key:
            raise ValueError("bucket and key are required for S3 upload")
        return _upload_s3_bytes(data, bucket, key, region)

    if dest == "local":
        if not dest_dir or not dest_name:
            raise ValueError("dest_dir and dest_name required for local upload")
        return await _upload_local_bytes(data, dest_dir, dest_name)

    raise ValueError("dest must be 's3' or 'local'")
