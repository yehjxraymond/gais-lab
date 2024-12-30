from config import llm_open_router, hf_embeddings, DATABASE_URL
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from typing import List
import json

# Setup for the vector store
vectorstore = PGVector(
    embeddings=hf_embeddings,
    collection_name="nais",
    connection=DATABASE_URL,
    use_jsonb=True,
)

@tool
def document_search_tool(query: str) -> List[str]:
    """
    Search for documents similar to the query in the vector store,
    and return top 5 matches.
    """
    # Use the vector store to search for similar documents
    results = vectorstore.similarity_search_with_score(query, 5)
    
    return [json.dumps({"metadata": doc.metadata}) + "\n" + doc.page_content for doc, _ in results]
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
   
    messages.append(SystemMessage("You have snippet of content from a report from Singapore National AI Strategy 2.0. Use information from the document to answer the question. Be sure to provide a detailed response and references."))
     
    # Final message to AI with the tool response
    final_response = llm_with_tools.invoke(messages)
    print("\n==== FINAL RESPONSE ====\n")
    print(final_response.content)

if __name__ == "__main__":
    main()
