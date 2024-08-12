import pandas as pd
from config import DATABASE_URL, hf_embeddings
from langchain_postgres import PGVector

if __name__ == "__main__":
    vectorstore = PGVector(
        embeddings=hf_embeddings,
        collection_name="nais",
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    # Searching documents
    results = vectorstore.similarity_search_with_score("venture funding for startups", 2)
    print(results)