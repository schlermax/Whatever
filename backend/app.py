import os
from pathlib import Path

from dotenv import load_dotenv

from services.ingestion.document_loader import load_text_documents
from services.ingestion.embeddings_model import embed_texts, get_cohere_embeddings
from services.ingestion.text_splitter import split_documents
from services.ingestion.vector_store import InMemoryVectorStore

# Load environment variables from .env file
load_dotenv()



def run_ingestion_pipeline(
    directory_path: str | Path | None = None,
    chunk_size: int = 100,
    chunk_overlap: int = 40,
    embedding_model: str = "embed-english-v3.0",
) -> InMemoryVectorStore:
    """Run the complete ingestion pipeline: load, chunk, embed, store."""
    root = Path(directory_path or Path(__file__).resolve().parent / "mock_data")
    print(f"Loading documents from: {root}")

    documents = load_text_documents(str(root), recursive=True)
    print(f"Loaded {len(documents)} documents\n")

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "<no source>")
        snippet = document.page_content.strip().replace("\n", " ")[:120]
        print(f"{index}. source={source}, length={len(document.page_content)}")
        print(f"   {snippet}{'...' if len(document.page_content) > 120 else ''}")

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"\nSplit into {len(chunks)} chunks\n")
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "<no source>")
        chunk_index = chunk.metadata.get("chunk_index", index)
        snippet = chunk.page_content.strip().replace("\n", " ")[:120]
        print(f"{index}. source={source}, chunk={chunk_index}, length={len(chunk.page_content)}")
        print(f"   {snippet}{'...' if len(chunk.page_content) > 120 else ''}")

    print(f"\nEmbedding chunks using Cohere model: {embedding_model}")
    embeddings_model = get_cohere_embeddings(model=embedding_model)
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_texts(embeddings_model, chunk_texts)
    print(f"Embedded {len(embeddings)} chunks\n")

    vector_store = InMemoryVectorStore()
    vector_store.add(chunks, embeddings)

    print(f"Vector Store: {vector_store}")
    print(f"Sample records:\n")
    for index, record in enumerate(vector_store.get_all()[:3], start=1):
        source = record.document.metadata.get("source", "<no source>")
        embedding_sample = record.embedding[:5]
        snippet = record.document.page_content.strip().replace("\n", " ")[:100]
        print(f"{index}. source={source}")
        print(f"   text: {snippet}...")
        print(f"   embedding (first 5 dims): {embedding_sample}")
        print()

    return vector_store


if __name__ == "__main__":
    run_ingestion_pipeline()
