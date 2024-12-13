from langchain_core.messages import HumanMessage, SystemMessage
from config import llm_open_router

def main():
    messages = [
        SystemMessage("You are a top notch comedian. For any topic that the user suggest give 10 jokes in the question punchline format. Format it as a JSON object with the key `jokes` and a list of jokes. Each joke should have a `question` and a `punchline`."),
        HumanMessage("Generative AI?")
    ]
    response = llm_open_router.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()
