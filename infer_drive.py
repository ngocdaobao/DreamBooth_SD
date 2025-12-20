from diffusers import StableDiffusionXLPipeline
import torch
import gdown

link = 'https://drive.google.com/file/d/1HQ5SoQqlghc4_WRI-7Vqy5x-22yigqls/view?usp=drive_link'
lora_path = gdown.download(link, 'pytorch_lora_weights.safetensors', quiet=False)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

pipe.load_lora_weights('pytorch_lora_weights.safetensors')
pipe.to('cuda')

torch.manual_seed(23)
image = pipe(prompt ='sks cat seen from the top', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('cat_top.png')