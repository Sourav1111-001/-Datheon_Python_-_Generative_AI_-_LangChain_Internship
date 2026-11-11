from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from modules.rag_pipeline import _message_text
from modules.vector_store import VectorStore


def generate_suggested_questions(
    chat_model: Any,
    store: VectorStore,
    document_id: str | None = None,
) -> list[str]:
    chunks = (
        store.get_document_chunks(document_id)
        if document_id
        else [doc for doc, _score in store.search("main topics and key ideas", top_k=8)]
    )
    if not chunks:
        raise ValueError("There is no indexed content to generate questions from.")
    context = "\n\n".join(chunk.page_content for chunk in chunks[:8])
    result = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "Create 5 to 8 useful, specific questions answerable from the supplied document text. "
                    "Return one question per line with no numbering, commentary, or answers."
                )
            ),
            HumanMessage(content=context),
        ]
    )
    questions = []
    for line in _message_text(result).splitlines():
        question = line.strip().lstrip("-*0123456789. )")
        if question and question not in questions:
            questions.append(question)
    if not questions:
        raise RuntimeError("The language model did not return any suggested questions.")
    return questions[:8]
