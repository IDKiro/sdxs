import gradio as gr
import spaces
from diffusers import StableDiffusionPipeline, AutoencoderKL
import os
import torch
from PIL import Image
import random

# SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", "0") == "1"

# Constants
repo = "IDKiro/sdxs-512-0.9"


# Ensure model and scheduler are initialized in GPU-enabled function
if torch.cuda.is_available():
    weight_type = torch.float32 
    pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type)

    # pipe.vae = AutoencoderKL.from_pretrained("IDKiro/sdxs-512-0.9/vae_large")     # use original VAE
    pipe.to("cuda")

# Function 
@spaces.GPU(enable_queue=True)
def generate_image(prompt):  
    seed  =  random.randint(-100000,100000)

    results =  pipe(
                prompt, 
                num_inference_steps=1, 
                guidance_scale=0,
                generator=torch.Generator(device="cuda").manual_seed(seed)
            )
    return results.images[0]



# Gradio Interface
description = """
This demo utilizes the SDXLS model of IDKiro/sdxs-512-0.9.

"""

with gr.Blocks(css="style.css") as demo:
    gr.HTML("<h1><center>Text-to-Image with SDXS ⚡</center></h1>")
    gr.Markdown(description)
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(label='Enter your prompt (English)', scale=8, value="portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour")
            submit = gr.Button(scale=1, variant='primary')
    img = gr.Image(label='SDXS Generated Image')

    prompt.submit(fn=generate_image,
                 inputs=[prompt],
                 outputs=img,
                 )
    submit.click(fn=generate_image,
                 inputs=[prompt],
                 outputs=img,
                 )
    
demo.queue().launch()