from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from modules.vector_store import VectorStore


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.lower().count("science")), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.lower().count("science")), 1.0]


def test_persistent_store_filters_and_deletes_document_vectors(tmp_path: Path) -> None:
    store = VectorStore(TinyEmbeddings(), tmp_path / "vectors")
    docs = [
        Document(
            page_content="Science explains the natural world.",
            metadata={"source": "science.txt", "document_id": "one", "chunk_id": "one-0"},
        ),
        Document(
            page_content="A separate text with other ideas.",
            metadata={"source": "other.txt", "document_id": "two", "chunk_id": "two-0"},
        ),
    ]
    store.add_documents(docs)
    filtered = store.search("science", document_id="one")
    assert len(filtered) == 1
    assert filtered[0][0].metadata["document_id"] == "one"

    reopened = VectorStore(TinyEmbeddings(), tmp_path / "vectors")
    assert len(reopened.get_document_chunks("one")) == 1
    reopened.delete_document("one")
    assert reopened.get_document_chunks("one") == []
    assert len(reopened.get_document_chunks("two")) == 1
