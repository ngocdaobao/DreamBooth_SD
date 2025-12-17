from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image


# Compute openpose conditioning image.
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(image)

# Initialize ControlNet pipeline.
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()


# Infer.
prompt = "Darth vader dancing in a desert, high quality"
negative_prompt = "low quality, bad quality"
images= pipe(
    prompt, 
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    num_images_per_prompt=4,
    image=openpose_image.resize((1024, 1024)),
    generator=torch.manual_seed(97),
).images[0]
image.save("openpose_controlnet_sdxl_inference.png")
