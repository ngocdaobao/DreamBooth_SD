from diffusers import StableDiffusionXLPipeline
import torch
import gdown

link = 'https://drive.google.com/drive/folders/1mKQ_uMf89ZK1swEb7QwAaUs0D1u2kv3F?usp=drive_link'
lora_path = gdown.download(link, 'pytorch_lora_weights.safetensors', quiet=False)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.load_lora_weights(lora_path)
pipe.to('cuda')

torch.manual_seed(23)
image = pipe(prompt ='sks cat seen from the top', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('cat_top.png')