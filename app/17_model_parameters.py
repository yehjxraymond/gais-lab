from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
llm_open_router = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1/",
    model="mistralai/mistral-7b-instruct:free", # Use models like mistral or phi-3
    # temperature=0.8, # range from 0 to 2
    # top_p=0.9, # range from 0 to 1
    # frequency_penalty=1, # range from -2 to 2
    # presence_penalty=1, # range from -2 to 2
)

response = llm_open_router.invoke("Tell me 3 jokes about generative AI")
print(response.content)