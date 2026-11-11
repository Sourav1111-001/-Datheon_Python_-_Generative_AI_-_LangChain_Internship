from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from modules.rag_pipeline import _message_text
from modules.vector_store import VectorStore


def summarize_document(
    chat_model: Any,
    store: VectorStore,
    document_id: str,
    detail: str = "concise",
    chunk_index: int | None = None,
) -> str:
    chunks = store.get_document_chunks(document_id)
    if not chunks:
        raise ValueError("No indexed content was found for this document.")
    if detail not in {"concise", "detailed"}:
        raise ValueError("Choose either a concise or detailed summary.")
    if chunk_index is not None:
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise ValueError("The selected document section is unavailable.")
        chunks = [chunks[chunk_index]]

    mapped: list[str] = []
    for start in range(0, len(chunks), 4):
        group = chunks[start : start + 4]
        passages = []
        for chunk in group:
            metadata = chunk.metadata
            citation = f"{metadata['source']}, page {metadata['page']}" if metadata.get("page") else metadata["source"]
            passages.append(f"Source: {citation}\n{chunk.page_content}")
        result = chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Summarize only the supplied document passages. Preserve important details and "
                        "distinguish facts from uncertainty. Do not add outside knowledge. Include source "
                        "filename and page citations for factual statements."
                    )
                ),
                HumanMessage(content="\n\n".join(passages)),
            ]
        )
        mapped.append(_message_text(result))

    level = "a detailed, structured summary" if detail == "detailed" else "a concise summary"
    final = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    f"Create {level} of the document using only these passage summaries. Also include key "
                    "topics, important definitions, and useful bullet-point notes. Retain source citations "
                    "where available. Do not invent information."
                )
            ),
            HumanMessage(content="\n\n".join(mapped)),
        ]
    )
    summary = _message_text(final)
    if not summary:
        raise RuntimeError("The language model returned an empty summary.")
    return summary
