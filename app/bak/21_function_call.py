import os

from config import llm_open_router
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
    tools = [search]
    llm_with_tools = llm_open_router.bind_tools(tools, tool_choice="search")
    response = llm_with_tools.invoke("What's the latest version of Langchain and what are the new features?")
    print(response)

if __name__ == "__main__":
    main()

# Tool Calling: https://python.langchain.com/v0.2/docs/how_to/tool_calling/
# Force calling: https://python.langchain.com/v0.2/docs/how_to/tool_choice/
