import os
import random
import time

import gradio as gr
import numpy as np
import PIL.Image
import torch

from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny
import torch

import os
import torch
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import uuid

device = "cuda"   # Linux & Windows
weight_type = torch.float16  # torch.float16 works as well, but pictures seem to be a bit worse

pipe = StableDiffusionPipeline.from_pretrained("IDKiro/sdxs-512-dreamshaper", torch_dtype=weight_type)
pipe.to(torch_device=device, torch_dtype=weight_type)

vae_tiny = AutoencoderTiny.from_pretrained("IDKiro/sdxs-512-dreamshaper/vae")
vae_large = AutoencoderKL.from_pretrained("IDKiro/sdxs-512-dreamshaper/vae_large")


def run(
    prompt: str,
    vae_type=None,
    param_dtype='torch.float16',
) -> PIL.Image.Image:
    if vae_type == "tiny vae":
        pipe.vae = vae_tiny
    elif vae_type == "large vae":
        pipe.vae = vae_large
    
    pipe.to(torch_device=device, torch_dtype=torch.float16 if param_dtype == 'torch.float16' else torch.float32)
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=1,
        output_type="pil",
    ).images

    print(time.time() - start_time)
    return result


examples = [
    "a close-up picture of an old man standing in the rain",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown("# SDXS-512-DreamShaper")
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", elem_id="output_image", show_label=False, show_download_button=True)
        
    with gr.Row():
        vae_choices = ['tiny vae','large vae']
        vae_type = gr.Radio(vae_choices, label='Image Decoder Type',  
                                    value=vae_choices[0],
                                    interactive=True,
                                    info='To save GPU memory, use tiny vae. For better quality, use large vae.')

        dtype_choices = ['torch.float16','torch.float32']
        param_dtype = gr.Radio(dtype_choices,label='torch.weight_type',  
                                    value=dtype_choices[0],
                                    interactive=True,
                                    info='To save GPU memory, use torch.float16. For better quality, use torch.float32.')

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=run
    )

    demo.load(None,None,None)

    inputs = [prompt, vae_type, param_dtype]
    outputs = [result]
    prompt.submit(fn=run, inputs=inputs, outputs=outputs)
    run_button.click(fn=run, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.queue().launch(debug=True)
