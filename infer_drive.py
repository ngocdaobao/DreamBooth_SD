from diffusers import StableDiffusionXLPipeline
import torch
import gdown



pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.load_lora_weights("dog2_dreambooth_model/pytorch_lora_weights.safetensors")
pipe.to('cuda')

torch.manual_seed(240)
image = pipe(prompt ='rwt dog wearing superman outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_superman.png')

image = pipe(prompt ='rwt dog wearing wizard outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_wizard.png')

image = pipe(prompt ='rwt dog wearing police outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_police.png')

image = pipe(prompt ='rwt dog wearing nurse outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_nurse.png')

image = pipe(prompt ='rwt dog wearing ironman outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_ironman.png')

image = pipe(prompt ='rwt dog wearing chef outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_chef.png')

image = pipe(prompt ='rwt dog wearing angel wings', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_angel.png')

image = pipe(prompt ='rwt dog wearing purple wizard outfit', num_inference_steps=40, guidance_scale=7.5).images[0]
image.save('dog_purple_wizard.png')