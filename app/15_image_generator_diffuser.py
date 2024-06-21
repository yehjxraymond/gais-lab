from diffusers import DiffusionPipeline

# You are not expected to run this script
# The script will download the SDXL model which is >10GB and the generation 
# will take a long time if you are running on CPU or a slow GPU
# This is just an example of how to use the DiffusionPipeline class works
# Make sure to run `poetry add diffusers` to install the required dependencies


pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# pipeline.to("cuda")

prompt = "a fairy queen riding a corgi into battle"

image = pipeline(prompt).images[0]

image.save(f"tmp/output_image.png")