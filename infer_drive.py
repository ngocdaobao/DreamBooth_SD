from diffusers import StableDiffusionXLPipeline
import torch
import gdown



pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.load_lora_weights("cat_dreambooth_model/pytorch_lora_weights.safetensors")
pipe.to('cuda')

torch.manual_seed(7)
image = pipe(prompt ='rwt cat is seen from the bottom view', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('cat_bottom.png')