from diffusers import StableDiffusionXLPipeline
import torch

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "dog_dreambooth_model"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.to("cuda")

# Load LoRA
pipe.load_lora_weights(lora_path)
image = pipe('a sks dog in strawberry farm', num_inference_steps=50).images[0]
image.save("inference_output.png")