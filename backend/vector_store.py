import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Define where your local Vector Database persistent files will be saved
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db_storage"
)


def get_vector_collection(collection_name: str = "legal_ease_docs"):
    """Initializes a persistent ChromaDB client and sets up an embedding function.

    Uses 'all-MiniLM-L6-v2' from Hugging Face which converts text to
    384-dimensional vectors.
    """
    # 1. Initialize persistent storage client so data saves to your disk
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Use a lightweight, popular open-source embedding model
    # This automatically downloads and runs the transformer model locally
    embedding_func = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    # 3. Create or fetch the collection (equivalent to a table in SQL)
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_func
    )
    return collection


def add_chunks_to_vector_db(chunks: list, source_name: str):
    """Converts text chunks to embeddings and inserts them into ChromaDB."""
    if not chunks:
        print("⚠️ No chunks provided to save.")
        return

    print(f"📦 Storing {len(chunks)} chunks into ChromaDB...")
    collection = get_vector_collection()

    # ChromaDB requires unique IDs, documents, and optional metadata strings
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name} for _ in range(len(chunks))]

    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    print(f"✅ Successfully indexed {len(chunks)} chunks in vector space.")


def query_vector_db(user_query: str, num_results: int = 3) -> list:
    """Performs a Cosine Similarity mathematical search against the Vector DB.

    Returns the top most semantically relevant text fragments.
    """
    print(f"🔍 Searching vector space for: '{user_query}'")
    collection = get_vector_collection()

    # The DB automatically encodes 'user_query' using the same embedding function
    # and evaluates proximity in multi-dimensional space
    results = collection.query(query_texts=[user_query], n_results=num_results)

    # Extract and clean up the matching documents list
    matched_documents = results.get("documents", [[]])[0]
    return matched_documents


if __name__ == "__main__":
    # Sample Mock Run to verify database functionality locally
    print("🚀 Initializing ChromaDB vector engine test run...")

    mock_chunks = [
        "The termination clause states a penalty fee of $5,000 applies if broken early.",
        "Intellectual property and source code remain 100% owned by the primary developer.",
        "This contract is governed exclusively by the laws and courts of California.",
    ]

    # Test Insertion
    add_chunks_to_vector_db(mock_chunks, source_name="test_contract.pdf")

    # Test Semantic Retrieval
    # Note that we don't use exact keywords like 'penalty' or '$5,000'
    test_search = "What happens if I cancel the agreement early?"
    matches = query_vector_db(test_search, num_results=1)

    print("\n--- Match Found ---")
    for idx, match in enumerate(matches):
        print(f"[{idx+1}] {match}")
