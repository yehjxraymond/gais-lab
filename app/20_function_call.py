from config import llm_open_router
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

# Note that you will have to use OpenAI's model for the tools to work
def main():
    tools = [search]
    llm_with_tools = llm_open_router.bind_tools(tools, tool_choice="duckduckgo_search")
    response = llm_with_tools.invoke("What's the weather in Singapore for the week?")
    print(response)

if __name__ == "__main__":
    main()

# Tool Calling: https://python.langchain.com/v0.2/docs/how_to/tool_calling/
# Force calling: https://python.langchain.com/v0.2/docs/how_to/tool_choice/
