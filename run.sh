# For original result:
#   --num_train_epochs=500
#   --num_class_images=200

# Define variables
class_token="backpack"
unique_token="sks"


python train_dreambooth_sdxl_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="dataset/${class_token}" \
  --class_data_dir="${class_token}_class_images" \
  --output_dir="${class_token}_dreambooth-model" \
  --instance_prompt="a ${unique_token} ${class_token}" \
  --class_prompt="a ${class_token}" \
  --num_class_images=2 \
  --class_token="${class_token}" \
  --unique_token="${unique_token}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --num_train_epochs=1 \
  --learning_rate=1e-4 \
  --mixed_precision="fp16" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --train_text_encoder \
  --use_8bit_adam \
  --gradient_checkpointing
