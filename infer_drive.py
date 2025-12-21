from diffusers import StableDiffusionXLPipeline
import torch
import gdown



pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.load_lora_weights("dog2_dreambooth_model/pytorch_lora_weights.safetensors")
pipe.to('cuda')

torch.manual_seed(100)
image = pipe(prompt ='rwt dog wearing chef outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog.png')