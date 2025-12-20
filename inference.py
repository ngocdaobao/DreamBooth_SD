from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
import torch
# import gdown
import argparse
import os
from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to the LoRA')
parser.add_argument('--base_model', type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help='Base model name or path')
parser.add_argument('--use_controlnet', action='store_true', help='Whether to use ControlNet')
parser.add_argument('--use_vae_finetuned', action='store_true', help='Whether to use finetuned VAE')
parser.add_argument('--class_token', type=str)
parser.add_argument('--controlnet_ckpt', type=str, default=None, help='Path to the ControlNet checkpoint')
parser.add_argument('--controlnet_condition_path', type=str, default=None, help='Path to the ControlNet conditioning image')
parser.add_argument('--vae_ckpt', type=str, default=None, help='Path to the VAE checkpoint')
parser.add_argument('--seed', type=int, default=42, help='Random seed for inference')
parser.add_argument('--num_inference_steps', type=int, default=40, help='Number of inference steps')
parser.add_argument('--unique_token', type=str, default='sks', help='Unique token used in DreamBooth training')
parser.add_argument('--num_per_prompt', type=int, default=4, help='Number of images to generate per prompt')

args = parser.parse_args()

unique_token = args.unique_token
class_token = args.class_token
prompt_list = {
'1':'a {0} {1} in the jungle'.format(unique_token, class_token),
'2':'a {0} {1} in the snow'.format(unique_token, class_token),
'3':'a {0} {1} on the beach'.format(unique_token, class_token),
'4':'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
'5':'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
'6':'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
'7':'a {0} {1} with a city in the background'.format(unique_token, class_token),
'8':'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
'9':'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
'10':'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
'11':'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
'12':'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
'13':'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
'14':'a {0} {1} floating on top of water'.format(unique_token, class_token),
'15':'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
'16':'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
'17':'a {0} {1} on top of a mirror'.format(unique_token, class_token),
'18':'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
'19':'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
'20':'a {0} {1} on top of a white rug'.format(unique_token, class_token),
'21':'a red {0} {1}'.format(unique_token, class_token),
'22':'a purple {0} {1}'.format(unique_token, class_token),
'23':'a shiny {0} {1}'.format(unique_token, class_token),
'24':'a wet {0} {1}'.format(unique_token, class_token),
'25':'a cube shaped {0} {1}'.format(unique_token, class_token),
# '26':'a {0} {1} pouring tea'.format(unique_token, class_token),
# '27':'a transparent {0} {1} within milk inside'.format(unique_token, class_token),
# '28':'a {0} {1} floating in the sea'.format(unique_token, class_token),
}

if args.use_controlnet:
    if args.controlnet_ckpt is None:
        raise ValueError("Please provide --controlnet_ckpt when using ControlNet")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_ckpt,
        torch_dtype=torch.float16,
    )

if args.use_vae_finetuned:
    if args.vae_ckpt is None:
        raise ValueError("Please provide --vae_ckpt when using finetuned VAE")
    vae = AutoencoderKL.from_single_file(
        args.vae_ckpt,
        torch_dtype=torch.float16,
    )

if args.use_controlnet and args.use_vae_finetuned:
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
elif args.use_controlnet:
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
elif args.use_vae_finetuned:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
else:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

pipeline.to('cuda')

# Load LoRA
pipeline.load_lora_weights(args.lora_ckpt, weight_name="pytorch_lora_weights.safetensors")

output_dir = f'{class_token}_inference_results'
os.makedirs(output_dir, exist_ok=True)

for idx, prompt in prompt_list.items():
    for i in range(args.num_per_prompt):
        set_seed = args.seed*(i+1)*200
        torch.manual_seed(set_seed)

        if args.use_controlnet:
            if args.controlnet_condition_path is None:
                raise ValueError("Please provide --controlnet_condition_path when using ControlNet")
            
        else:
            result = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.5,
            ).images[0]

        result.save(f"{output_dir}/{idx}_{i}.png") 





