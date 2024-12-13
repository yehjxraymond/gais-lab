import streamlit as st
from langchain_postgres import PGVector
from langchain_core.documents import Document
from config import DATABASE_URL, hf_embeddings
import psycopg2

# Initialize PGVector
vectorstore = PGVector(
    embeddings=hf_embeddings,
    collection_name="sample_articles",
    connection=DATABASE_URL,
    use_jsonb=True,
)

# Initialize PostgreSQL connection
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

page_size = 10
search_result_size = 2

def list_documents(limit=page_size, offset=0):
    cur.execute("SELECT id, document FROM langchain_pg_embedding ORDER BY id LIMIT %s OFFSET %s", (limit, offset))
    return cur.fetchall()

def count_documents():
    cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
    return cur.fetchone()[0]

def add_document(content):
    doc = Document(page_content=content)
    vectorstore.add_documents([doc])

def delete_document(doc_id):
    vectorstore.delete([doc_id])

def search_documents(query, limit=search_result_size):
    return vectorstore.similarity_search_with_score(query, limit)

def display_document(content, doc_id, key_prefix="document"):
    if len(content) > 200:
        with st.expander(f"{content[:200]}... Click to expand"):
            st.write(content)
    else:
        st.write(content)
    if doc_id != None and st.button("Delete", key=f"delete_{key_prefix}_{doc_id}"):
        delete_document(doc_id)
        st.rerun()

def main():
    st.title("Document Management with Vector Store")

    # Pagination setup
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    # Calculate offset based on current page
    offset = st.session_state.current_page * page_size

    # List documents with pagination
    st.subheader("Documents")
    documents = list_documents(limit=page_size, offset=offset)
    for doc_id, content in documents:
        display_document(content, doc_id, key_prefix="list")

    # Pagination controls
    col1, col2 = st.columns(2)
    # Calculate total pages
    total_documents = count_documents()
    total_pages = (total_documents + page_size - 1) // page_size

    with col1:
        # Show "Previous" button only if not on the first page
        if st.session_state.current_page > 0:
            if st.button("Previous"):
                st.session_state.current_page -= 1
                st.rerun()

    with col2:
        # Show "Next" button only if not on the last page
        if st.session_state.current_page < total_pages - 1:
            if st.button("Next"):
                st.session_state.current_page += 1
                st.rerun()

    # Search documents
    st.subheader("Search Documents")
    query = st.text_input("Enter search query")
    if st.button("Search"):
        results = search_documents(query)
        for [document, score] in results:
            st.write(f"Score: {score:.2f}")
            display_document(document.page_content, None, key_prefix="search")

    # Add document
    st.subheader("Add Document")
    new_doc = st.text_area("Enter document content")
    if st.button("Save"):
        add_document(new_doc)
        st.success("Document added!")
        st.rerun()

if __name__ == "__main__":
    main()
    
    
# If you are facing the issue of adjustText not found, run the app with the full command
# `python3 -m streamlit run app/13_vector_store_streamlit.py`
