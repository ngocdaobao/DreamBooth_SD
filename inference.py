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
image = pipe('a photo of sks girl in Paris street, clear face, with the head slightly tilted forward; the left arm extended forward and outward with a bent elbow, the right arm bent at the elbow and positioned closer to the torso; the right leg bent at the knee and positioned forward, and the left leg bent at the knee and positioned backward.', num_inference_steps=40).images[0]
image.save("pose_prompt_output.png")
