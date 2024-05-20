from config import llm_open_router

def main():
    response = llm_open_router.invoke("Say hello world in a pirate voice!")
    print(response.content)

if __name__ == "__main__":
    main()
