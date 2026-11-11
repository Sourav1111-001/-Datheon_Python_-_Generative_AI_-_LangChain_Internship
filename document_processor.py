from __future__ import annotations

import hashlib
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


class DocumentProcessingError(ValueError):
    """Raised when an uploaded document cannot be safely indexed."""


@dataclass(frozen=True)
class ProcessedDocument:
    document_id: str
    filename: str
    file_type: str
    content_hash: str
    page_count: int | None
    chunks: list[Document]


def sanitize_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    base = PurePosixPath(normalized).name
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if safe in {"", ".", ".."}:
        raise DocumentProcessingError("The uploaded filename is invalid.")
    return safe


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_upload(
    filename: str,
    content: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ProcessedDocument:
    safe_name = sanitize_filename(filename)
    suffix = PurePosixPath(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise DocumentProcessingError("Only PDF and TXT files are supported.")
    if not content:
        raise DocumentProcessingError("The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise DocumentProcessingError(f"Files must be {MAX_FILE_SIZE_MB} MB or smaller.")
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise DocumentProcessingError("Chunk overlap must be smaller than the positive chunk size.")

    document_id = str(uuid.uuid4())
    file_type = suffix[1:]
    content_hash = hashlib.sha256(content).hexdigest()
    pages: list[Document] = []
    try:
        with tempfile.TemporaryDirectory(prefix="documind-") as temporary_directory:
            path = Path(temporary_directory) / f"upload{suffix}"
            path.write_bytes(content)
            if suffix == ".pdf":
                loaded_pages = PyPDFLoader(str(path), mode="page").load()
                page_count: int | None = len(loaded_pages)
                for page in loaded_pages:
                    text = _clean_text(page.page_content)
                    if text:
                        metadata = {
                            "source": safe_name,
                            "file_type": file_type,
                            "document_id": document_id,
                            "page": int(page.metadata["page"]) + 1,
                        }
                        pages.append(Document(page_content=text, metadata=metadata))
            else:
                try:
                    loaded_pages = TextLoader(str(path), encoding="utf-8-sig").load()
                except UnicodeDecodeError as exc:
                    raise DocumentProcessingError("TXT files must use UTF-8 encoding.") from exc
                text = _clean_text(loaded_pages[0].page_content) if loaded_pages else ""
                if text:
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={"source": safe_name, "file_type": file_type, "document_id": document_id},
                        )
                    )
                page_count = None
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError("The file is corrupted or could not be read.") from exc

    if not pages:
        raise DocumentProcessingError("No readable text was found in the document.")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{document_id}-{index:06d}"
    if not chunks:
        raise DocumentProcessingError("No searchable text chunks could be created.")
    return ProcessedDocument(document_id, safe_name, file_type, content_hash, page_count, chunks)
