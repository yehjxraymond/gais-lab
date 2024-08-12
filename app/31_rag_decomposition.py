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
    and return top 5 matches.
    """
    # Use the vector store to search for similar documents
    results = vectorstore.similarity_search_with_score(query, 5)
    
    return [json.dumps({"metadata": doc.metadata}) + "\n" + doc.page_content for doc, _ in results]
def main():
    query = "What resources are available to AI Startups in Singapore?"
    
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
Generate multiple search queries related to: {question}
Format your output to a list of 3 queries, one query per line. Your output should only be 3 lines long."""
    
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    generate_queries_decomposition = ( prompt_decomposition | llm_open_router | StrOutputParser() | (lambda x: x.split("\n")))
    questions = generate_queries_decomposition.invoke({"question":query})

    print("=============== Decomposed Questions =================")    
    print(questions);
    print("=============== Decomposed Questions =================")    
    
    # Initialize a list to hold all documents for all questions
    all_documents = []
    
    # Iterate over each decomposed question to retrieve 3 documents
    for i, decomposed_question in enumerate(questions):
        documents = document_search_tool(decomposed_question)
        all_documents.append({"question": decomposed_question, "documents": documents})
        print (f"=============== Documents for Question {i+1} =================")
        print(documents)
        print (f"=============== Documents for Question {i+1} =================")
    
    # Construct a new prompt for the LLM to generate a comprehensive reply
    llm_prompt = f"Original Query: {query}\n\n"
    for item in all_documents:
        llm_prompt += f"Decomposed Question: {item['question']}\n"
        llm_prompt += "Documents:\n"
        for doc in item['documents']:
            llm_prompt += f"- {doc}\n"
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
