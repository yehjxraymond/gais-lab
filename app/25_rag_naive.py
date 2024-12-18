import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from config import llm_open_router, hf_embeddings

NAIS_PICKLE = "./fixtures/nais.pkl"

def find_top_k_similar(embedding, embeddings, num_results=5):
    """
    Finds the top k most similar documents to the given embedding.
    
    :param embedding: The query embedding vector.
    :param embeddings: Numpy array of document embeddings.
    :param num_results: The number of top results to return (default is 5).
    :return: Indices of the top k similar documents and their similarities.
    """
    similarities = [1 - cosine(embedding, doc_emb) for doc_emb in embeddings]
    top_k_indices = np.argsort(similarities)[-num_results:]  # Get indices of top k results
    return [(index, similarities[index]) for index in reversed(top_k_indices)]  # Return in descending order of similarity

def main():
    ## START: Code from previous step ##
    df = pd.read_pickle(NAIS_PICKLE)
    documents = df['content'].tolist()
    metadata = df['source'].tolist()
    document_embeddings = np.array(df['embedding'].tolist())
    sample_query = "What resources are available to AI Startups in Singapore?"
    query_embedding = hf_embeddings.embed_query(sample_query)
    print(f"Query: {sample_query}")
    print(f"Query embedding: {query_embedding[:5]}\n")
    top_results = find_top_k_similar(query_embedding, document_embeddings, num_results=5)
    for index, similarity in top_results:
        print(f"Document {index + 1}: {documents[index]}")
        print(f"Similarity: {similarity}\n")    
    ## END: Code from previous step ##
    
    # Preparing the message for the AI
    top_results_formatted = [
        f"<document>\nPage: {metadata[index]['page']+1}\nContent: {documents[index]}\n</document>"
        for index, _ in top_results
    ]
    message = f"""
    You are a helpful assistant that can answer questions about the Singapore National AI Strategy 2.0. You have snippets of content from the report to assist you in crafting the response. Use information from the snippet to answer the question. Be sure to provide a detailed response and cite the source page number in the response when you use the information. Provide specific information from the document whenever possible.
    
    Resources:
    {'\n'.join(top_results_formatted)}
    
    Question:
    {sample_query}
    """
    print(message)
    
    final_response = llm_open_router.invoke(message)
    print("\n==== FINAL RESPONSE ====\n")
    print(final_response.content)

if __name__ == "__main__":
    main()