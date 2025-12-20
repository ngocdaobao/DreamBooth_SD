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
DEFAULT_UNIQUE="sks"

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

python inference.py \
    --lora_ckpt="${class_token}_dreambooth_model/pytorch_lora_weights.safetensors" \
    --class_token="${class_token}" \
    --unique_token="${unique_token}" 
 

