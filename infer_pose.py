import torch
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)
from diffusers.utils import load_image
from PIL import Image

# =====================
# Paths
# =====================
lora_weight_path = "girl_dreambooth_model/pytorch_lora_weights.safetensors"
pose_image_path = "poses/standing_03.png"

prompt = "a photo of girl on the beach"

# =====================
# Load pose image
# =====================
pose_image = Image.open(pose_image_path).convert("RGB")
pose_image = pose_image.resize((1024, 1024))

# =====================
# Load T2I Adapter (OpenPose SDXL)
# =====================
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# =====================
# Load SDXL Adapter Pipeline
# =====================
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    adapter=adapter,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe.load_lora_weights(lora_weight_path)
pipe.to("cuda")


# =====================
# Inference
# =====================
result = pipe(
    prompt=prompt,
    image=pose_image,
    adapter_conditioning_scale=0.8,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

result.save("infer_pose_output.png")
