import random
import numpy as np
from PIL import Image
import base64
from io import BytesIO

import torch
import torchvision.transforms.functional as F
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import gradio as gr

device = "cuda"   # Linux & Windows
weight_type = torch.float16  # torch.float16 works as well, but pictures seem to be a bit worse

controlnet = ControlNetModel.from_pretrained(
    "IDKiro/sdxs-512-dreamshaper-sketch", torch_dtype=weight_type
).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "IDKiro/sdxs-512-dreamshaper", controlnet=controlnet, torch_dtype=weight_type
)
pipe.to("cuda")

style_list = [
    {
        "name": "No Style",
        "prompt": "{prompt}",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
    },
]

styles = {k["name"]: k["prompt"] for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "No Style"
MAX_SEED = np.iinfo(np.int32).max


def pil_image_to_data_url(img, format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def run(
    image, 
    prompt, 
    prompt_template, 
    style_name, 
    controlnet_conditioning_scale,
    device_type="GPU",
    param_dtype='torch.float16',
):
    if device_type == "CPU":
        device = "cpu" 
        param_dtype = 'torch.float32'
    else:
        device = "cuda"
    
    pipe.to(torch_device=device, torch_dtype=torch.float16 if param_dtype == 'torch.float16' else torch.float32)

    print(f"prompt: {prompt}")
    print("sketch updated")
    if image is None:
        ones = Image.new("L", (512, 512), 255)
        temp_url = pil_image_to_data_url(ones)
        return ones, gr.update(link=temp_url), gr.update(link=temp_url)
    prompt = prompt_template.replace("{prompt}", prompt)
    control_image = image.convert("RGB")
    control_image = Image.fromarray(255 - np.array(control_image))

    output_pil = pipe(
        prompt=prompt,
        image=control_image,
        width=512,
        height=512,
        guidance_scale=0.0,
        num_inference_steps=1,
        num_images_per_prompt=1,
        output_type="pil",
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]

    input_sketch_url = pil_image_to_data_url(control_image)
    output_image_url = pil_image_to_data_url(output_pil)
    return (
        output_pil,
        gr.update(link=input_sketch_url),
        gr.update(link=output_image_url),
    )


def update_canvas(use_line, use_eraser):
    if use_eraser:
        _color = "#ffffff"
        brush_size = 20
    if use_line:
        _color = "#000000"
        brush_size = 8
    return gr.update(brush_radius=brush_size, brush_color=_color, interactive=True)


def upload_sketch(file):
    _img = Image.open(file.name)
    _img = _img.convert("L")
    return gr.update(value=_img, source="upload", interactive=True)


scripts = """
async () => {
    globalThis.theSketchDownloadFunction = () => {
        console.log("test")
        var link = document.createElement("a");
        dataUrl = document.getElementById('download_sketch').href
        link.setAttribute("href", dataUrl)
        link.setAttribute("download", "sketch.png")
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up
      
        // also call the output download function
        theOutputDownloadFunction();
      return false
    }

    globalThis.theOutputDownloadFunction = () => {
        console.log("test output download function")
        var link = document.createElement("a");
        dataUrl = document.getElementById('download_output').href
        link.setAttribute("href", dataUrl);
        link.setAttribute("download", "output.png");
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up
      return false
    }

    globalThis.UNDO_SKETCH_FUNCTION = () => {
        console.log("undo sketch function")
        var button_undo = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(1)');
        // Create a new 'click' event
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        button_undo.dispatchEvent(event);
    }

    globalThis.DELETE_SKETCH_FUNCTION = () => {
        console.log("delete sketch function")
        var button_del = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(2)');
        // Create a new 'click' event
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        button_del.dispatchEvent(event);
    }

    globalThis.togglePencil = () => {
        el_pencil = document.getElementById('my-toggle-pencil');
        el_pencil.classList.toggle('clicked');
        // simulate a click on the gradio button
        btn_gradio = document.querySelector("#cb-line > label > input");
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        btn_gradio.dispatchEvent(event);
        if (el_pencil.classList.contains('clicked')) {
            document.getElementById('my-toggle-eraser').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
        else {
            document.getElementById('my-toggle-eraser').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
        
    }

    globalThis.toggleEraser = () => {
        element = document.getElementById('my-toggle-eraser');
        element.classList.toggle('clicked');
        // simulate a click on the gradio button
        btn_gradio = document.querySelector("#cb-eraser > label > input");
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        btn_gradio.dispatchEvent(event);
        if (element.classList.contains('clicked')) {
            document.getElementById('my-toggle-pencil').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
        else {
            document.getElementById('my-toggle-pencil').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
    }
}
"""

with gr.Blocks(css="style.css") as demo:
    # these are hidden buttons that are used to trigger the canvas changes
    line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
    eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")
    with gr.Row(elem_id="main_row"):
        with gr.Column(elem_id="column_input"):
            gr.Markdown("## INPUT", elem_id="input_header")
            image = gr.Image(
                source="canvas", tool="color-sketch", type="pil", image_mode="L",
                invert_colors=True, shape=(512, 512), brush_radius=8, height=440, width=440,
                brush_color="#000000", interactive=True, show_download_button=True, elem_id="input_image", show_label=False)
            download_sketch = gr.Button("Download sketch", scale=1, elem_id="download_sketch")
            
            gr.HTML("""
            <div class="button-row">
                <div id="my-div-pencil" class="pad2"> <button id="my-toggle-pencil" onclick="return togglePencil(this)"></button> </div>
                <div id="my-div-eraser" class="pad2"> <button id="my-toggle-eraser" onclick="return toggleEraser(this)"></button> </div>
                <div class="pad2"> <button id="my-button-undo" onclick="return UNDO_SKETCH_FUNCTION(this)"></button> </div>
                <div class="pad2"> <button id="my-button-clear" onclick="return DELETE_SKETCH_FUNCTION(this)"></button> </div>
                <div class="pad2"> <button href="TODO" download="image" id="my-button-down" onclick='return theSketchDownloadFunction()'></button> </div>
            </div>
            """)
            # gr.Markdown("## Prompt", elem_id="tools_header")
            prompt = gr.Textbox(label="Prompt", value="", show_label=True)
            with gr.Row():
                style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME, scale=1)
                prompt_temp = gr.Textbox(label="Prompt Style Template", value=styles[DEFAULT_STYLE_NAME], scale=2, max_lines=1)
            
            controlnet_conditioning_scale = gr.Slider(label="Control Strength", minimum=0, maximum=1, step=0.01, value=0.8)

                 
            device_choices = ['GPU','CPU']
            device_type = gr.Radio(device_choices, label='Device',  
                                        value=device_choices[0],
                                        interactive=True,
                                        info='Please choose GPU if you have a GPU.')
            
            dtype_choices = ['torch.float16','torch.float32']
            param_dtype = gr.Radio(dtype_choices,label='torch.weight_type',  
                                        value=dtype_choices[0],
                                        interactive=True,
                                        info='To save GPU memory, use torch.float16. For better quality, use torch.float32.')
                

        with gr.Column(elem_id="column_process", min_width=50, scale=0.4):
            gr.Markdown("## SDXS-Sketch", elem_id="description")
            run_button = gr.Button("Run", min_width=50)

        with gr.Column(elem_id="column_output"):
            gr.Markdown("## OUTPUT", elem_id="output_header")
            result = gr.Image(label="Result", height=440, width=440, elem_id="output_image", show_label=False, show_download_button=True)
            download_output = gr.Button("Download output", elem_id="download_output")
            gr.Markdown("### Instructions")
            gr.Markdown("**1**. Enter a text prompt (e.g. cat)")
            gr.Markdown("**2**. Start sketching")
            gr.Markdown("**3**. Change the image style using a style template")
            gr.Markdown("**4**. Adjust the effect of sketch guidance using the slider")

    
    eraser.change(fn=lambda x: gr.update(value=not x), inputs=[eraser], outputs=[line]).then(update_canvas, [line, eraser], [image])
    line.change(fn=lambda x: gr.update(value=not x), inputs=[line], outputs=[eraser]).then(update_canvas, [line, eraser], [image])

    demo.load(None,None,None,_js=scripts)
    inputs = [image, prompt, prompt_temp, style, controlnet_conditioning_scale, device_type, param_dtype]
    outputs = [result, download_sketch, download_output]
    prompt.submit(fn=run, inputs=inputs, outputs=outputs)
    style.change(lambda x: styles[x], inputs=[style], outputs=[prompt_temp]).then(
        fn=run, inputs=inputs, outputs=outputs,)
    run_button.click(fn=run, inputs=inputs, outputs=outputs)
    image.change(run, inputs=inputs, outputs=outputs,)

if __name__ == "__main__":
    demo.queue().launch(debug=True)