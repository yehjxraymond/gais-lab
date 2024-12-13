# https://api.python.langchain.com/en/latest/vectorstores/langchain_postgres.vectorstores.PGVector.html#langchain_postgres.vectorstores.PGVector
# You may have to add the pgvector extension to your database. You can do this by running the following command in your database:
# CREATE EXTENSION IF NOT EXISTS vector;
from langchain_core.messages import HumanMessage, SystemMessage
from config import DATABASE_URL, hf_embeddings

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

import psycopg2


vectorstore = PGVector(
    embeddings=hf_embeddings,
    collection_name="sample_articles",
    connection=DATABASE_URL,
    use_jsonb=True,
)

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

def main():
    # Adding document
    docs = [
        Document(
            page_content="there are cats in the pond",
        ),
        Document(
            page_content="there's a pot of gold at the end of the rainbow",
        )
    ]
    results = vectorstore.add_documents(docs)
    
    # Delete document
    # vectorstore.delete(["document-id-here"])
    
    
    # List documents
    cur.execute("SELECT id, document FROM langchain_pg_embedding")
    results = cur.fetchall()
    print(results)
    
    # Searching documents
    results = vectorstore.similarity_search_with_score("meow", 2)
    print(results)
    



if __name__ == "__main__":
    main()
