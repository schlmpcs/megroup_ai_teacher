from pathlib import Path

from fastapi import HTTPException, UploadFile, status


async def read_upload(
    file: UploadFile,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum size of {max_bytes} bytes",
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return b"".join(chunks)


async def stream_upload(
    file: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> int:
    total = 0
    with destination.open("wb") as output:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum size of {max_bytes} bytes",
                )
            output.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return total
