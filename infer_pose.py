import gdown
import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel
)
from diffusers.utils import load_image

# =====================
# Download LoRA
# =====================
gdown.download_folder(
    "https://drive.google.com/drive/folders/1h50l1oRCSXTppUmm2oZXCOwvmT7BBAIV",
    output="ckpt_for_infer_pose",
    quiet=False
)

lora_weight_path = "ckpt_for_infer_pose/teapot_dreambooth_model/teapot_dreambooth_model/pytorch_lora_weights.safetensors"
# image_for_pose = "home-page-slider-3.jpg"

# =====================
# OpenPose (SDXL compatible)
# =====================
# pose_detector = OpenposeDetector.from_pretrained(
#     "lllyasviel/ControlNet"
# )

# image = load_image(image_for_pose)
# pose_image = pose_detector(image)
# pose_image = pose_image.resize((1024, 1024))

# =====================
# ControlNet SDXL (QUAN TRỌNG)
# =====================
# controlnet = ControlNetModel.from_pretrained(
#     "thibaud/controlnet-openpose-sdxl-1.0",
#     torch_dtype=torch.float16
# )

# =====================
# SDXL ControlNet Pipeline
# =====================
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe.load_lora_weights(lora_weight_path)
pipe.to("cuda")

# =====================
# Inference
# =====================
result = pipe('a sks teapot with the Eiffel Tower in the background', num_inference_steps=50).images[0]

result.save("infer_pose_output.png")
