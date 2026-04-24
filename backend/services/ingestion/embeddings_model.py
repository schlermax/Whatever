from typing import Iterable

from langchain_cohere import CohereEmbeddings


def get_cohere_embeddings(model: str = "embed-english-v3.0") -> CohereEmbeddings:
    return CohereEmbeddings(model=model)


def embed_texts(embeddings: CohereEmbeddings, texts: Iterable[str]) -> list[list[float]]:
    return embeddings.embed_documents(list(texts))


def embed_query(embeddings: CohereEmbeddings, query: str) -> list[float]:
    return embeddings.embed_query(query)


__all__ = ["get_cohere_embeddings", "embed_texts", "embed_query"]