from diffusers.utils.torch_utils import randn_tensor

num_steps = 30
pose_start_ratio = 0.6   # pose starts after 60%
pose_scale = 0.6

pipe.scheduler.set_timesteps(num_steps, device="cuda")

# 1. Encode prompt
prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
    prompt=prompt,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

# 2. Prepare latents
latents = randn_tensor(
    (1, pipe.unet.config.in_channels, 128, 128),
    device="cuda",
    dtype=torch.float16,
)

# 3. Encode pose ONCE
adapter_states = pipe.adapter(
    image=pose_image,
    device="cuda",
    dtype=torch.float16,
)

# 4. Denoising loop with late pose
for i, t in enumerate(pipe.scheduler.timesteps):

    # 🔑 late pose gating
    if i < int(num_steps * pose_start_ratio):
        cur_adapter_scale = 0.0
    else:
        cur_adapter_scale = pose_scale

    latent_model_input = torch.cat([latents] * 2)

    noise_pred = pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        pooled_encoder_hidden_states=pooled_prompt_embeds,
        adapter_states=adapter_states,
        adapter_conditioning_scale=cur_adapter_scale,
    ).sample

    noise_uncond, noise_text = noise_pred.chunk(2)
    noise_pred = noise_uncond + 7.5 * (noise_text - noise_uncond)

    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# 5. Decode
image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
image = pipe.image_processor.postprocess(image)[0]

image.save("infer_pose_output.png")
