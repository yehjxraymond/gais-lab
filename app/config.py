from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
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

# Hugging Face Inference API
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGING_FACE_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
