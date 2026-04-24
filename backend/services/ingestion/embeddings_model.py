from typing import Iterable

from langchain_cohere import CohereEmbeddings


def get_cohere_embeddings(model: str = "embed-english-v3.0") -> CohereEmbeddings:
    """Initialize and return a CohereEmbeddings instance.

    Args:
        model: The Cohere model to use for embeddings.

    Returns:
        A CohereEmbeddings instance ready to embed text.

    Note:
        Requires COHERE_API_KEY environment variable to be set or passed directly.
    """
    return CohereEmbeddings(model=model)


def embed_texts(embeddings: CohereEmbeddings, texts: Iterable[str]) -> list[list[float]]:
    """Embed a collection of texts using the provided embeddings model.

    Args:
        embeddings: A CohereEmbeddings instance.
        texts: Iterable of text strings to embed.

    Returns:
        A list of embedding vectors (list of floats).
    """
    return embeddings.embed_documents(list(texts))


def embed_query(embeddings: CohereEmbeddings, query: str) -> list[float]:
    """Embed a single query string.

    Args:
        embeddings: A CohereEmbeddings instance.
        query: The query text to embed.

    Returns:
        An embedding vector (list of floats).
    """
    return embeddings.embed_query(query)


__all__ = ["get_cohere_embeddings", "embed_texts", "embed_query"]