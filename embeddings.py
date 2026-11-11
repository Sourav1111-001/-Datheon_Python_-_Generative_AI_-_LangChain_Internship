from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL


def create_embeddings(model_name: str = EMBEDDING_MODEL) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
