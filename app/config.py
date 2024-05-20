from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Local LLM
llm_local = ChatOpenAI(
    api_key="NIL",
    openai_api_base="http://localhost:1234/v1/",
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

# Openrouter LLM
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
llm_open_router = ChatOpenAI(
    api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1/"
)
