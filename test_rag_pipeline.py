from types import SimpleNamespace

from langchain_core.documents import Document

from modules.rag_pipeline import NO_ANSWER, RagPipeline


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, top_k=4, document_id=None):
        self.queries.append((query, top_k, document_id))
        return self.results


class FakeModel:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return SimpleNamespace(content=self.answers.pop(0))


def test_no_relevant_sources_returns_exact_fallback_without_llm_answer() -> None:
    model = FakeModel(["unused"])
    pipeline = RagPipeline(FakeStore([]), model)
    result = pipeline.answer("What is it?")
    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert model.prompts == []


def test_answer_uses_retrieved_context_and_returns_real_source_metadata() -> None:
    store = FakeStore(
        [
            (
                Document(
                    page_content="Photosynthesis converts light into chemical energy.",
                    metadata={"source": "biology.pdf", "page": 7, "document_id": "doc"},
                ),
                0.82,
            )
        ]
    )
    model = FakeModel(["Photosynthesis converts light into chemical energy. [S1]"])
    result = RagPipeline(store, model).answer("What is photosynthesis?", top_k=2)
    assert result.answer.endswith("[S1]")
    assert result.sources == [
        {
            "filename": "biology.pdf",
            "page": 7,
            "snippet": "Photosynthesis converts light into chemical energy.",
        }
    ]
    assert store.queries == [("What is photosynthesis?", 2, None)]


def test_followup_history_rewrites_search_query_without_answering() -> None:
    store = FakeStore([])
    model = FakeModel(["What risks does photosynthesis have?", "answer not used"])
    result = RagPipeline(store, model).answer(
        "What are its risks?",
        history=[{"role": "assistant", "content": "We discussed photosynthesis."}],
    )
    assert result.answer == NO_ANSWER
    assert store.queries[0][0] == "What risks does photosynthesis have?"
