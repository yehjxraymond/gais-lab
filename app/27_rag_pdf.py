from config import llm_open_router, hf_embeddings
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from typing import List

NAIS_PICKLE = "./fixtures/nais.pkl"

@tool
def document_search_tool(query: str) -> List[str]:
    """
    Search for documents similar to the query by loading them from a pickle file,
    and return top 5 matches.
    """
    df = pd.read_pickle(NAIS_PICKLE)
    documents = df['content'].tolist()
    document_embeddings = np.array(df['embedding'].tolist())
    
    query_embedding = hf_embeddings.embed_query(query)
    top_results = find_top_k_similar(query_embedding, document_embeddings, num_results=5)
    
    return [documents[index] for index, _ in top_results]

def find_top_k_similar(embedding, embeddings, num_results=5):
    from scipy.spatial.distance import cosine
    similarities = [1 - cosine(embedding, doc_emb) for doc_emb in embeddings]
    top_k_indices = np.argsort(similarities)[-num_results:]
    return [(index, similarities[index]) for index in reversed(top_k_indices)]

def main():
    query = "What resources are available to AI Startups in Singapore?"
    messages = [HumanMessage(query)]

    tools = [document_search_tool]  # Use the custom document search tool
    llm_with_tools = llm_open_router.bind_tools(tools)
    
    # Initial message to AI to construct the tool calls
    ai_message = llm_with_tools.invoke(messages)
    print(ai_message.tool_calls)
    messages.append(ai_message)
    
    # Assuming tool calls are needed, calling the document_search_tool with the query
    if ai_message.tool_calls:
        tool_msg = document_search_tool.invoke(query)  # Directly use the query since the tool only needs this parameter
        tool_response = " ".join(tool_msg)  # Construct a single string from the top documents
        print(tool_response)
        messages.append(ToolMessage(tool_response, tool_call_id=ai_message.tool_calls[0]["id"]))
    
    # Final message to AI with the tool response
    final_response = llm_with_tools.invoke(messages)
    print("\n==== FINAL RESPONSE ====\n")
    print(final_response.content)

if __name__ == "__main__":
    main()