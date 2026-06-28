from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create/Open ChromaDB
client = chromadb.PersistentClient(path="vector_db")


def get_collection():
    return client.get_or_create_collection(
        name="financial_documents"
    )


def clear_collection():

    try:
        client.delete_collection("financial_documents")
    except Exception:
        pass

    return client.get_or_create_collection(
        name="financial_documents"
    )


def store_embeddings(chunks, file_name):

    collection = get_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        ids.append(f"{file_name}_{i}")
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append(
            {
                "source": file_name,
                "chunk": i + 1
            }
        )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(chunks)