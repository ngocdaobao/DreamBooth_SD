from diffusers import StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLPipeline
from diffusers import ControlNetModel
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import gdown



# lora_ckpt = gdown.download(
#     "https://drive.google.com/file/d/1W39qIrpJzyJk-gIa2-TsN0_MQG5Jcq4U/view",
#     output="lora_ckpt.safetensors",
#     fuzzy=True
# )

lora_ckpt = 'girl_dreambooth_model/pytorch_lora_weights.safetensors'

vae_path = 'girl_dreambooth_model/vae_finetuned/diffusion_pytorch_model.safetensors'
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)

vae = AutoencoderKL.from_single_file(
    vae_path,
    torch_dtype=torch.float16
).to("cuda")

# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     controlnet=controlnet,
#     # vae=vae,
#     torch_dtype=torch.float16,
# )

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.load_lora_weights(lora_ckpt)
pipe.to('cuda')
pose = 'poses/dance_01.png'
pose_image = Image.open(pose)
pose_image = pose_image.resize((1024,1024))
# prompt = 'a rwt girl in Paris street with this pose: Full-body human figure, dynamic walking pose, front view, upright torso with slight forward lean, head slightly tilted to the right. Left arm extended sideways at shoulder height with a slight elbow bend, right arm bent inward with forearm relaxed downward. Legs crossed mid-stride: left leg stepping forward across the body, knee slightly bent; right leg extended backward with foot lifted. Sense of motion and balance, natural anatomy, realistic proportions, smooth pose transition.'
prompt = 'a rwt girl in Paris street'
negative_prompt = 'identity drift, blurry, low quality, cartoon, unrealistic'
torch.manual_seed(30)

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

result.save("pose_prompt.png")
