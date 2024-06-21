import requests
from config import HUGGING_FACE_API_KEY
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

image_bytes = query({
	"inputs": "a fairy queen riding a corgi into battle",
	"parameters": {
		# "negative_prompt": "wings, crown, scepter, ugly, malformed",
     	# "width": 512,
        # "height": 512,
		# "seed": 1 # For determinism
	},
})

image = Image.open(io.BytesIO(image_bytes))
image.save("tmp/output_image.png")