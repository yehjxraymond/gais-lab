import pandas as pd
from config import DATABASE_URL, hf_embeddings
from langchain_postgres import PGVector

NAIS_PICKLE = "./fixtures/nais.pkl"

def load_data_to_vectorstore():
    # Load DataFrame from pickle file
    df = pd.read_pickle(NAIS_PICKLE)
    
    # Extract documents and their embeddings
    documents = df['content'].tolist()
    document_embeddings = df['embedding'].tolist()
    
    # Initialize PGVector
    vectorstore = PGVector(
        embeddings=hf_embeddings,
        collection_name="nais",
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    # Prepare metadata for each document with source information
    metadatas = [{'source': df['source'][i]['source'], 'page': df['source'][i]['page']} for i in range(len(documents))]
    
    # Add documents and embeddings to the vectorstore
    vectorstore.add_embeddings(texts=documents, embeddings=document_embeddings, metadatas=metadatas)

if __name__ == "__main__":
    load_data_to_vectorstore()