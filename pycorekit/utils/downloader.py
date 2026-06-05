"""
Generic Document Downloader Utility

Supports:
- HTTP/HTTPS downloads
- S3 downloads
- Local file reads
- Streaming large files
- Automatic temp file creation

Usage:
    from pycorekit.utils.downloader import download_file

    path = await download_file("https://example.com/file.pdf")
"""

import os
import tempfile
import aiohttp
import boto3
from urllib.parse import urlparse
from typing import Optional


# ---------------------------------------------------------
# HTTP / HTTPS DOWNLOAD
# ---------------------------------------------------------

async def _download_http(url: str) -> str:
    """Download a file from HTTP/HTTPS and return local temp file path."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()

            suffix = os.path.splitext(urlparse(url).path)[1]
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)

            with open(temp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 64):
                    f.write(chunk)

            return temp_path


# ---------------------------------------------------------
# S3 DOWNLOAD
# ---------------------------------------------------------

def _download_s3(url: str, region: str = "us-east-1") -> str:
    """
    Download a file from S3 using boto3.
    URL format: s3://bucket/key/path/file.pdf
    """
    parsed = urlparse(url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3", region_name=region)

    suffix = os.path.splitext(key)[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    s3.download_file(bucket, key, temp_path)
    return temp_path


# ---------------------------------------------------------
# LOCAL FILE READ
# ---------------------------------------------------------

def _download_local(path: str) -> str:
    """Return absolute path for local files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    return os.path.abspath(path)


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

async def download_file(url: str, s3_region: str = "us-east-1") -> str:
    """
    Download a file from:
    - HTTP/HTTPS
    - S3 (s3://bucket/key)
    - Local path

    Returns:
        str: Local file path
    """
    if url.startswith("http://") or url.startswith("https://"):
        return await _download_http(url)

    if url.startswith("s3://"):
        return _download_s3(url, region=s3_region)

    return _download_local(url)
