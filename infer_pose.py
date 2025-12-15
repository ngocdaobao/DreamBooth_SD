import gdown
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

gdown.download_folder(
    "https://drive.google.com/drive/folders/1h50l1oRCSXTppUmm2oZXCOwvmT7BBAIV?usp=drive_link",
    output = 'ckpt_for_infer_pose',
    quiet=False
)

lora_weight_path = "ckpt_for_infer_pose/pytorch_lora_weights.safetensors"
image_for_pose = "home-page-slider-3.jpg"

pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)

image = load_image(image_for_pose)
pose_condition = pose_detector(image)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

pipe.load_lora_weights(lora_weight_path)

pipe.to("cuda")

image = pipe(
    prompt="a sks dog in the beach",
    image=pose_condition.resize((1024, 1024)),
    num_inference_steps=40
).images[0]

image.save("infer_pose_output.png")

