from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="vector_db")


def search_documents(query, selected_pdf="All Documents", top_k=3):

    collection = client.get_or_create_collection(
        name="financial_documents"
    )

    query_embedding = model.encode(query).tolist()

    if selected_pdf == "All Documents":

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    else:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={
                "source": selected_pdf
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    return results