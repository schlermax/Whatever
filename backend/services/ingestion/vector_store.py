from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document


@dataclass
class VectorRecord:
    """A single record in the vector store containing a document and its embedding."""

    document: Document
    embedding: list[float]


class InMemoryVectorStore:
    """Simple in-memory vector store for storing documents with their embeddings."""

    def __init__(self):
        self.records: list[VectorRecord] = []

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents with their embeddings to the store.

        Args:
            documents: List of LangChain Document objects.
            embeddings: List of embedding vectors corresponding to documents.

        Raises:
            ValueError: If document and embedding counts don't match.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        for document, embedding in zip(documents, embeddings):
            self.records.append(VectorRecord(document=document, embedding=embedding))

    def count(self) -> int:
        """Return the number of records in the store."""
        return len(self.records)

    def get_all(self) -> list[VectorRecord]:
        """Return all records in the store."""
        return self.records

    def get_record(self, index: int) -> Optional[VectorRecord]:
        """Get a single record by index."""
        if 0 <= index < len(self.records):
            return self.records[index]
        return None

    def clear(self) -> None:
        """Clear all records from the store."""
        self.records = []

    def __repr__(self) -> str:
        return f"InMemoryVectorStore(records={len(self.records)})"


__all__ = ["VectorRecord", "InMemoryVectorStore"]