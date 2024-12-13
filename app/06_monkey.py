from langchain_core.messages import HumanMessage, SystemMessage
from config import llm_open_router

def main():
    messages = [
        SystemMessage("You are a monkey. Respond to all query using only noises made by monkeys."),
        HumanMessage("What are the use cases of generative AI?")
    ]
    response = llm_open_router.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()
