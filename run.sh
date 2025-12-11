# For orginal result:
#   --num_train_epochs=500
#   --num_class_images=200

python train_dreambooth_sdxl_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="dataset/backpack" \
  --class_data_dir="/kaggle/input/class_images" \
  --output_dir="/kaggle/working/dreambooth-model" \
  --instance_prompt="a sks backpack" \
  --class_prompt="a backpack" \
  --num_class_images=2 \
  --class_token="backpack" \
  --unique_token="sks" \
  --resolution=1024 \
  --train_batch_size=1 \
  --num_train_epochs=1 \ 
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --mixed_precision="fp16" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --train_text_encoder \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing 