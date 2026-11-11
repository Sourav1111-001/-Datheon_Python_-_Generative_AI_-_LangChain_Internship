from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import CHROMA_DIR


class VectorStore:
    def __init__(self, embeddings: Embeddings, persist_directory: Path = CHROMA_DIR) -> None:
        self._embeddings = embeddings
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name="documind_documents",
            embedding_function=embeddings,
            persist_directory=str(self._persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Cannot index an empty document.")
        ids = [str(doc.metadata["chunk_id"]) for doc in documents]
        self._store.add_documents(documents=documents, ids=ids)

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        filters: dict[str, Any] | None = {"document_id": document_id} if document_id else None
        return self._store.similarity_search_with_relevance_scores(query, k=top_k, filter=filters)

    def get_document_chunks(self, document_id: str) -> list[Document]:
        result = self._store.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        chunks = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(result["documents"] or [], result["metadatas"] or [], strict=True)
        ]
        return sorted(chunks, key=lambda doc: doc.metadata.get("chunk_id", ""))

    def delete_document(self, document_id: str) -> None:
        self._store.delete(where={"document_id": document_id})

    def clear(self) -> None:
        self._store.delete_collection()
        self._store = Chroma(
            collection_name="documind_documents",
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )
