from config import llm_open_router, hf_embeddings, DATABASE_URL
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from typing import List
import json
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate



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
    and return top 10 matches.
    """
    # Use the vector store to search for similar documents
    results = vectorstore.similarity_search_with_score(query, 10)
    
    return [json.dumps({"metadata": doc.metadata}) + "\n" + doc.page_content for doc, _ in results]

def main():
    query = "What resources are available to AI Startups in Singapore?"
    
    template = """You are a helpful assistant that generates hypothetical responses to user queries about the Singapore National AI Strategy report.
The goal is provide a comprehensive response to the user query by breaking it down into sub-questions and providing detailed answers.
Provide a response directly the the user query without commenting on the process or mentioning it's a hypothetical response.

User Query: {question}
"""
    
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    hyde = ( prompt_decomposition | llm_open_router | StrOutputParser() )
    hypothetical_response = hyde.invoke({"question":query})

    print("=============== Hypothetical Response =================")    
    print(hypothetical_response);
    print("=============== Hypothetical Response =================")    
    
    documents = document_search_tool(hypothetical_response)
    
    # Construct a new prompt for the LLM to generate a comprehensive reply
    llm_prompt = f"Original Query: {query}\n\n"
    for i, document in enumerate(documents):
        llm_prompt += f"Document {i}:\n"
        llm_prompt += f"{document}\n"
        llm_prompt += "\n"
    
    llm_prompt += "Based on the above information, compose a comprehensive reply to the original query. Use information from the document to answer the question. Be sure to provide a detailed response and references."
    
    print("\n==== LLM PROMPT ====\n")
    print(llm_prompt)
    print("\n==== LLM PROMPT ====\n")
    
    # Invoke the LLM with the new prompt
    comprehensive_reply = llm_open_router.invoke(llm_prompt)
    
    print("\n==== COMPREHENSIVE REPLY ====\n")
    print(comprehensive_reply.content)

if __name__ == "__main__":
    main()
    
