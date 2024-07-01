from langsmith import traceable
from config import llm_open_router

# Sample: https://smith.langchain.com/public/24dbb7f8-ae5d-4332-aa91-bd7e76729cbe/r

@traceable
def main():
    response = llm_open_router.invoke("Say hello world in a pirate voice!")
    print(response.content)

if __name__ == "__main__":
    main()
