<div align="center">

## SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions

[![Project](https://img.shields.io/badge/Project-forestgreen?logo=Houzz&logoColor=white)](https://idkiro.github.io/sdxs)
[![Paper](https://img.shields.io/badge/Paper-blue?logo=arxiv)](https://arxiv.org/abs/2403.16627) 
[![SDXS-512-0.9](https://img.shields.io/badge/🤗SDXS--512--0.9-darkorange)](https://huggingface.co/IDKiro/sdxs-512-0.9)
[![SDXS-512-DreamShaper](https://img.shields.io/badge/🤗SDXS--512--DreamShaper-darkorange)](https://huggingface.co/IDKiro/sdxs-512-dreamshaper)
[![SDXS-512-DreamShaper](https://img.shields.io/badge/🤗SDXS--512--DreamShaper--Sketch-darkorange)](https://huggingface.co/IDKiro/sdxs-512-dreamshaper-sketch/tree/main)

*Yuda Song, Zehao Sun, Xuanwu Yin*

</div>

We present two models, SDXS-512 and SDXS-1024, achieving inference speeds of approximately <b>100 FPS</b> (30x faster than SD v1.5) and <b>30 FPS</b> (60x faster than SDXL) on a single GPU. Assuming the image generation time is limited to <b>1 second</b>, then SDXL can only use 16 NFEs to produce a slightly blurry image, while SDXS-1024 can generate 30 clear images. 

![](images/intro.png)

Moreover, our proposed method can also train ControlNet, offering promising applications in image-conditioned control and facilitating efficient image-to-image translation.

<p align="left" >
<img src="images\sketch.gif" width="800" />
</p>

## News

- **April 10, 2024:** [SDXS-512-DreamShaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper) and [SDXS-512-DreamShaper-Sketch](https://huggingface.co/IDKiro/sdxs-512-dreamshaper-sketch) is released. We also upload our demo code.
- **March 25, 2024:** [SDXS-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9) is released, it is an old version of SDXS-512.

## Usage

Install requirements:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers accelerate diffusers==0.25.1 gradio==3.43.1
```

Run text-to-image demo:

```
python demo.py
```

Run sketch-to-image demo:

```
python demo_sketch.py
```

## Method

### Model Acceleration

We train an extremely light-weight image decoder to mimic the original VAE decoder’s output through a combination of output distillation loss and GAN loss. We also leverage the block removal distillation strategy to efficiently transfer the knowledge from the original U-Net to a more compact version.

![](images/method1.png)

SDXS demonstrates efficiency far surpassing that of the base models, even achieving image generation at 100 FPS for 512x512 images and 30 FPS for 1024x1024 images on the GPU.

![](images/speed.png)

### Text-to-Image

To reduce the NFEs, we suggest straightening the sampling trajectory and quickly finetuning the multi-step model into a one-step model by replacing the distillation loss function with the proposed feature matching loss. Then, we extend the Diff-Instruct training strategy, using the gradient of the proposed feature matching loss to replace the gradient provided by score distillation in the latter half of the timestep.

![](images/method2.png)

Despite a noticeable downsizing in both the sizes of the models and the number of sampling steps required, the prompt-following capability of SDXS-512 remains superior to that of SD v1.5. This observation is consistently validated in the performance of SDXS-1024 as well.  

![](images/imgs.png)

### Image-to-Image

We extend our proposed training strategy to the training of ControlNet, relying on adding the pretrained ControlNet to the score function. 

![](images/method3.png)

We demonstrate its efficacy in facilitating image-to-image conversions utilizing ControlNet, specifically for transformations involving canny edges and depth maps.

![](images/control_imgs.png)


## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{song2024sdxs,
  author    = {Yuda Song, Zehao Sun, Xuanwu Yin},
  title     = {SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions},
  journal   = {arxiv},
  year      = {2024},
}
```

**Acknowledgment**: the demo code is based on https://github.com/GaParmar/img2img-turbo.
