from diffusers import StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import ControlNetModel
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import gdown



vae_path = "girl_dreambooth_model/vae/diffusion_pytorch_model.safetensors"
lora_ckpt = 'girl_dreambooth_model/pytorch_lora_weights.safetensors'
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
print("Load VAE from:", vae_path)
vae = AutoencoderKL.from_single_file(
    vae_path,
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.load_lora_weights(lora_ckpt,)
pipe.to('cuda')
pose = 'poses/dance_01.png'
pose_image = Image.open(pose)
pose_image = pose_image.resize((1024,1024))
prompt = 'a rwt girl in Paris street, high resolution'
negative_prompt = 'identity drift, blurry, low quality'
torch.manual_seed(32)
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=pose_image,
    num_inference_steps=40,
    guidance_scale=7.5,

    controlnet_conditioning_scale=1.0,
    control_guidance_start=0.0,   # 🔑 start pose late
    control_guidance_end=1.0,
).images[0]

result.save("pose_face_loss.png")
