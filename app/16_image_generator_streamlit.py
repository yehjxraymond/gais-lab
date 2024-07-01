import random
import streamlit as st
import requests
from PIL import Image
import io
from config import HUGGING_FACE_API_KEY

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def main():
    st.title("Image Generator")

    prompt = st.text_input("Enter a prompt for the image:", "")
    negative_prompt = st.text_input("Enter a negative prompt (optional):", "")

    with st.expander("Advanced Adjustments"):
        deterministic_toggle = st.checkbox("Use deterministic seed", value=False)
        seed = st.number_input("Seed (for determinism):", min_value=0, value=1, step=1, disabled=not deterministic_toggle)
        width = st.number_input("Width:", min_value=64, value=512, step=64)
        height = st.number_input("Height:", min_value=64, value=512, step=64)

    generate_button = st.button("Generate Image")

    if generate_button and prompt:
        parameters = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt if negative_prompt else None,
                "width": width,
                "height": height,
            },
        }
        if deterministic_toggle:
            parameters["parameters"]["seed"] = seed
        else :
            parameters["parameters"]["seed"] = random.randint(1, 2**32)
        image_bytes = query(parameters)
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Generated Image")

if __name__ == "__main__":
    main()
    
# If you are facing the issue of adjustText not found, run the app with the full command
# `python3 -m streamlit run app/16_image_generator_streamlit.py`
