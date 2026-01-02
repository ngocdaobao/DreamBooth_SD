#!/usr/bin/env bash
set -euo pipefail

# For original result:
#   --num_train_epochs=500
#   --num_class_images=200

# This script accepts `class_token` via flag or positional arg.
# `unique_token` remains a default and is not overridden by the CLI.
# Usage examples:
#   ./run.sh -c backpack
#   ./run.sh backpack
#   ./run.sh            # uses defaults

# Defaults
DEFAULT_UNIQUE="rwt"

unique_token="$DEFAULT_UNIQUE"

usage() {
  echo "Usage: $0 [-c class_token] [class_token]"
  exit 1
}

# Parse flags
while getopts ":c:h" opt; do
  case ${opt} in
    c ) class_token="$OPTARG" ;;
    h ) usage ;;
    \? ) echo "Invalid option: -$OPTARG" 1>&2; usage ;;
    : ) echo "Invalid option: -$OPTARG requires an argument" 1>&2; usage ;;
  esac
done

shift $((OPTIND -1))

# Allow positional arg to override class_token after flags
if [ "$#" -ge 1 ]; then
  class_token="$1"
fi


echo "Using class_token=${class_token}, unique_token=${unique_token} (default)"

python train_dreambooth_pose.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="dataset/${class_token}" \
  --class_data_dir="${class_token}_class_images" \
  --output_dir="${class_token}_dreambooth_model" \
  --instance_prompt="a ${unique_token} ${class_token}" \
  --class_prompt="a ${class_token}" \
  --num_class_images=100 \
  --class_token="${class_token}" \
  --unique_token="${unique_token}" \
  --resolution=1024 \
  --with_prior_preservation \
  --train_batch_size=1 \
  --max_train_steps=1000 \
  --learning_rate=5e-5 \
  --mixed_precision="fp16" \
  --prior_loss_weight=0.5 \
  --use_8bit_adam \
  --gradient_checkpointing 
 

