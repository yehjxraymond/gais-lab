import os

from config import llm_open_router
from langchain_core.messages import HumanMessage, ToolMessage
from exa_py import Exa
from langchain_core.tools import tool
exa = Exa(api_key=os.environ["EXA_API_KEY"])

@tool
def search(query: str):
    """Search for webpages based on the query and retrieve their contents."""
    # This combines two API endpoints: search and contents retrieval
    return exa.search_and_contents(
        query, use_autoprompt=True, num_results=5, text=True, highlights=True
    )

# Note that you will have to use OpenAI's model for the tools to work
def main():
    query = "What's the latest version of Langchain and what are the new features?"
    messages = [HumanMessage(query)]

    tools = [search]
    llm_with_tools = llm_open_router.bind_tools(tools)
    
    # Initial message to AI to construct the tool calls
    ai_message = llm_with_tools.invoke(messages)
    print(ai_message.tool_calls)
    messages.append(ai_message)

    # Calling the tools with the parameter from the AI message
    tool_msg = search.invoke(ai_message.tool_calls[0]["args"])
    print(tool_msg)
    messages.append(ToolMessage(tool_msg, tool_call_id=ai_message.tool_calls[0]["id"]))
    
    # # Final message to AI with the tool response
    # final_response = llm_with_tools.invoke(messages)
    # print(final_response.content)






if __name__ == "__main__":
    main()

# Tool Calling: https://python.langchain.com/v0.2/docs/how_to/tool_calling/
# Force calling: https://python.langchain.com/v0.2/docs/how_to/tool_choice/
