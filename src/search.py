from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="vector_db")


def search_documents(query, top_k=25):
    """
    Search the ChromaDB vector database and return the
    most relevant document chunks.
    """

    # Load existing collection
    collection = client.get_or_create_collection(
        name="financial_documents"
    )

    # Convert query to embedding
    query_embedding = model.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results