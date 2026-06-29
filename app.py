import streamlit as st
import os

from src.pdf_parser import extract_text
from src.text_processor import split_text
from src.embedding import store_embeddings, clear_collection
from src.search import search_documents
from src.bedrock_llm import generate_answer

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Enterprise Financial Intelligence Assistant",
    page_icon="📊",
    layout="wide"
)
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
# ---------------------------------
# Title
# ---------------------------------
st.title("📊 Enterprise Financial Intelligence Assistant")
st.write("Enterprise Financial Intelligence Assistant using RAG")

with st.sidebar:

    st.title("Enterprise BI Assistant")

    st.markdown("---")

    st.success("✅ Amazon Bedrock")
    st.success("✅ ChromaDB")
    st.success("✅ Semantic Search")
    st.success("✅ RAG")

    st.markdown("---")

    st.write("### Instructions")

    st.write("1. Upload PDF files")
    st.write("2. Wait for processing")
    st.write("3. Ask questions")
# ==========================================
# SEARCH SECTION
# ==========================================

st.header("💬 Ask Questions")

import chromadb

# Read uploaded PDF names from ChromaDB
client = chromadb.PersistentClient(path="vector_db")

try:
    collection = client.get_collection("financial_documents")

    data = collection.get()

    pdf_names = sorted(
        list(
            set(
                meta["source"]
                for meta in data["metadatas"]
            )
        )
    )

except:
    pdf_names = []

selected_pdf = st.selectbox(
    "📄 Select Document",
    ["All Documents"] + pdf_names
)

question = st.text_input(
    "Ask a question about your uploaded documents"
)

if st.button("Search"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        try:

            # Search selected document or all documents
            results = search_documents(
                question,
                selected_pdf
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            context = ""

            for doc, meta in zip(documents, metadatas):

                context += f"""

Source: {meta["source"]}

{doc}

--------------------------------------------------------

"""

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    question,
                    context
                )

            st.subheader("🤖 AI Answer")

            st.success(answer)

            st.subheader("📚 Sources")

            unique_sources = sorted(
                set(
                    meta["source"]
                    for meta in metadatas
                )
            )

            for source in unique_sources:
                st.write(f"• {source}")

        except Exception as e:

            st.error(str(e))
# ==========================================
# UPLOAD SECTION
# ==========================================

st.divider()

st.header("📂 Upload Financial Documents")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.documents_processed:
    clear_collection()

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            uploaded_file.name
        )

        # Save PDF
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )

        # Extract text
        text = extract_text(file_path)

        # Split text
        chunks = split_text(text)

        st.success(
            f"Total Chunks Created: {len(chunks)}"
        )

        # Store embeddings
        stored = store_embeddings(
            chunks,
            uploaded_file.name
        )

        st.success(
            f"{stored} embeddings stored successfully!"
        )

        # Preview
        st.subheader(
            f"📄 Preview - {uploaded_file.name}"
        )

        for i, chunk in enumerate(chunks[:3]):

            with st.expander(f"Chunk {i+1}"):

                st.write(chunk)
    st.session_state.documents_processed = True

if st.button("🔄 Upload New Documents"):
    st.session_state.documents_processed = False
    st.rerun()

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Enterprise Financial Intelligence Assistant | RAG + ChromaDB"
)