from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="vector_db")


def search_documents(query, top_k=3):

    # Get the latest collection every time
    collection = client.get_or_create_collection(
        name="financial_documents"
    )

    # Convert query to embedding
    query_embedding = model.encode(query).tolist()

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results