from diffusers import StableDiffusionXLPipeline
import torch
import gdown

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
link = "https://drive.google.com/file/d/1gNjI7LlcSdlJwa50Dw9p9o8wgJkfcrOH/view?usp=drive_link"
lora_ckpt = gdown.download(link, quiet=False, fuzzy=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.to("cuda")

# Load LoRA
torch.manual_seed(2)
pipe.load_lora_weights(lora_ckpt, weight_name="pytorch_lora_weights.safetensors")
image = pipe('a photo of sks girl in Paris street, the head tilted forward about 10°, one arm extended forward at roughly 40° while the other is bent backward around 30°, one leg angled forward about 30° with a slight knee bend, and the other leg angled backward about 35° with a deeper knee bend.', 
             num_inference_steps=40).images[0]
image.save("pose_prompt_output.png")
