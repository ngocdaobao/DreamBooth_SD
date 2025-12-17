from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import ControlNetModel
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import gdown

link = "https://drive.google.com/file/d/1gNjI7LlcSdlJwa50Dw9p9o8wgJkfcrOH/view?usp=drive_link"


lora_ckpt = gdown.download(link, quiet=False, fuzzy=True)
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.load_lora_weights(lora_ckpt, weight_name="pytorch_lora_weights.safetensors")
pipe.to('cuda')
pose = 'poses/dance_01.png'
pose_image = Image.open(pose)
pose_image = pose_image.resize((1024,1024))
prompt = 'a photo of sks girl in Paris street'
# negative_prompt = 'distorted face, NOT sks girl, blurred, low quality, ugly'
torch.manual_seed(2)
result = pipe(
    prompt=prompt,
    # negative_prompt=negative_prompt,
    image=pose_image,
    num_inference_steps=40,
    guidance_scale=7.5,

    controlnet_conditioning_scale=0.8,
    control_guidance_start=0.6,   # 🔑 start pose late
    control_guidance_end=1.0,
).images[0]

result.save("late_pose_08_output.png")