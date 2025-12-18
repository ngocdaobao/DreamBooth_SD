#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from email import generator
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
# import clip

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import torch.utils.checkpoint
import transformers
# import timm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from insightface.app import FaceAnalysis

# from metrics import im2im, im2prompt

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    ControlNetModel,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_peft_version,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.36.0.dev0")

logger = get_logger(__name__)


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    vae_path=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# {"SDXL" if "playground" not in base_model else "Playground"} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    if "playground" in base_model:
        model_description += """\n
## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++" if "playground" not in base_model else "playground-v2dot5-community",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora" if not use_dora else "dora",
        "template:sd-lora",
    ]
    if "playground" in base_model:
        tags.extend(["playground", "playground-diffusers"])
    else:
        tags.extend(["stable-diffusion-xl", "stable-diffusion-xl-diffusers"])

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if not args.do_edm_style_training:
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
    # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--unique_token", type=str, default='sks')
    parser.add_argument("--class_token", type=str, default=None)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        action="append",
        help="Prompt(s) to use during validation. Can be specified multiple times for multiple prompts.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=45,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://huggingface.co/papers/2206.00364.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
        help="Flag to additionally generate final state dict in the Kohya format so that it becomes compatible with A111, Comfy, Kohya, etc.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Whether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://huggingface.co/papers/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )

    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )
 #New args
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--pred_image_dir",
        type=str,
        default="pred_images_finetune_vae",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []

        interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {interpolation=}.")
        train_resize = transforms.Resize(size, interpolation=interpolation)

        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        original_size = self.original_sizes[index % self.num_instance_images]
        crop_top_left = self.crop_top_lefts[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        original_sizes += [example["original_size"] for example in examples]
        crop_top_lefts += [example["crop_top_left"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only interested in the pooled output of the final text encoder (text_encoder_two)
        if i == len(text_encoders) - 1:
            pooled_prompt_embeds = prompt_embeds[0]
        
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds



def main(args):
    # ...existing code...
    # Save VAE checkpoint after training
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        subfolder="pipeline",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_auth_token=True,
    )

    pipeline.load_lora_weights(args.lora_weights)
    pipeline.to("cuda")

    unet = pipeline.unet
    text_encoder_one = pipeline.text_encoder
    text_encoder_two = pipeline.text_encoder_2
    vae = pipeline.vae

    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(False)

    for name, param in vae.named_parameters():
        if "decoder" in name:
            param.requires_grad = True

    pose_path = 'poses'
    poses = []
    for img in os.listdir(pose_path):
        pose = load_image(os.path.join(pose_path, img))
        pose = pose.resize((1024,1024))
        pose_transform = T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),   # converts PIL → Tensor in [0,1]
        ])
        pose = pose_transform(pose)  # Remove unsqueeze(0) - DataLoader will add batch dimension
        poses.append(pose)
    
    class PoseDataset(Dataset):
        def __init__(self, poses):
            self.poses = poses

        def __len__(self):
            return len(self.poses)

        def __getitem__(self, idx):
            return self.poses[idx]

    prompt = args.instance_prompt


    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if hasattr(args, 'mixed_precision') else None,
        log_with=args.report_to if hasattr(args, 'report_to') else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.logging_dir)),
    )

    # Use 8-bit AdamW optimizer from bitsandbytes
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            filter(lambda p: p.requires_grad, vae.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    except ImportError:
        raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
    num_train_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_train_steps,
        power=args.lr_power,
        num_cycles=args.lr_num_cycles,
    )
    pose_dataset = PoseDataset(poses)
    pose_dataloader = DataLoader(
        pose_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    # Prepare models, optimizer, dataloader with accelerator
    vae, optimizer, pose_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, pose_dataloader, lr_scheduler
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Extract prompt embeddings using the encode_prompt function
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    prompt_hidden_states, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
    prompt_hidden_states = prompt_hidden_states.to("cuda").half()
    pooled_prompt_embeds = pooled_prompt_embeds.to("cuda").half()
    empty_prompt_hidden_states, pooled_empty_prompt_embeds = encode_prompt(
        text_encoders, tokenizers, ""
    )
    empty_prompt_hidden_states = empty_prompt_hidden_states.to("cuda").half()
    pooled_empty_prompt_embeds = pooled_empty_prompt_embeds.to("cuda").half()
    
    # Debug: Print shapes immediately after encoding
    print("\n=== PROMPT ENCODING DEBUG ===")
    print(f"prompt_hidden_states shape: {prompt_hidden_states.shape}")
    print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
    print(f"empty_prompt_hidden_states shape: {empty_prompt_hidden_states.shape}")
    print(f"pooled_empty_prompt_embeds shape: {pooled_empty_prompt_embeds.shape}")
    print("============================\n")

    # Debug: print shapes to understand what we're working with
    print(f"prompt_hidden_states shape: {prompt_hidden_states.shape}")
    print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
    
    face_encoder = FaceAnalysis(
        name="buffalo_l",  # or mobilenetv2-glint360k
        providers=["CUDAExecutionProvider"]
    )

    face_encoder = FaceAnalysis('buffalo_l')
    face_encoder.prepare(ctx_id=0, det_thresh=0.6, det_size=(512, 512))

    def extract_face_embedding(img) -> torch.tensor: #detect face in img as np array then extract embedding of face
        faces = face_encoder.get(img)
        if len(faces) == 0:
            return None
        face = faces[0]
        embedding = face.embedding
        embedding = torch.tensor(embedding, dtype=torch.float32)
        return embedding
       
    
    #Crop face of instance images
    face_embeddings = []
    for face in os.listdir(args.instance_data_dir):
        face_path = os.path.join(args.instance_data_dir, face)
        face_img = Image.open(face_path).convert('RGB')
        face_np = np.array(face_img)[:, :, ::-1]  # RGB to BGR for insightface
        emb = extract_face_embedding(face_np)
        if emb is None:
            raise ValueError(f"No face detected in image {face_path}")
        face_embeddings.append(emb)
    face_embeddings = torch.stack(face_embeddings).to('cuda')
    mean_face_embedding = torch.mean(face_embeddings, dim=0, keepdim=True)

     #Compute mean embedding of cropped faces


    progress_bar = tqdm(range(0, args.max_train_steps), desc="Training VAE Decoder")
    global_step = 0
    pipeline.scheduler.set_timesteps(
        num_inference_steps=30,
        device="cuda"
    )

    for epoch in range(args.num_train_epochs):
        vae.train()
        for step, pose_batch in enumerate(pose_dataloader):
            with accelerator.accumulate(vae):
                latent_shape = (pose_batch.shape[0], 4, 1024//8, 1024//8)
                latent = torch.randn(latent_shape, device=pose_batch.device)
                print(len(pipeline.scheduler.timesteps))
                for i, t in enumerate(pipeline.scheduler.timesteps):
                    latent_model_input = pipeline.scheduler.scale_model_input(latent, t).half()
                    with torch.no_grad():
                    # correct pooled embeds for ControlNet
                        # Repeat embeddings to match batch size if needed
                        batch_size = latent_model_input.shape[0]
                        controlnet_text_embeds = pooled_empty_prompt_embeds.repeat(batch_size, 1)
                        controlnet_time_ids = torch.zeros(
                            (batch_size, 6),
                            dtype=pooled_empty_prompt_embeds.dtype,
                            device=pooled_empty_prompt_embeds.device,
                        )

                        # Debug: print what we're passing
                        if i == 0 and step == 0:
                            print(f"\n=== CONTROLNET INPUT DEBUG (Step {step}, Timestep {i}) ===")
                            print(f"batch_size: {batch_size}")
                            print(f"controlnet_text_embeds shape: {controlnet_text_embeds.shape}")
                            print(f"controlnet_text_embeds dtype: {controlnet_text_embeds.dtype}")
                            print(f"controlnet_time_ids shape: {controlnet_time_ids.shape}")
                            print(f"controlnet_time_ids dtype: {controlnet_time_ids.dtype}")
                            print(f"Expected concat dim: {controlnet_text_embeds.shape[-1]} + {controlnet_time_ids.shape[-1]} = {controlnet_text_embeds.shape[-1] + controlnet_time_ids.shape[-1]}")
                            print("=" * 60)
                        
                        controlnet_added_cond = {
                            "text_embeds": controlnet_text_embeds,
                            "time_ids": controlnet_time_ids
                        }
                    
                        down_res, mid_res = controlnet(
                            latent_model_input,
                            t,
                            controlnet_cond=pose_batch.half(),
                            encoder_hidden_states=empty_prompt_hidden_states.repeat(batch_size, 1, 1),
                            added_cond_kwargs=controlnet_added_cond,
                            return_dict=False
                        )
                        
                        # UNet also needs added_cond_kwargs for SDXL
                        unet_text_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
                        unet_time_ids = torch.zeros(
                            (batch_size, 6),
                            dtype=pooled_prompt_embeds.dtype,
                            device=pooled_prompt_embeds.device,
                        )
                        
                        unet_added_cond = {
                            "text_embeds": unet_text_embeds,
                            "time_ids": unet_time_ids
                        }
                        
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_hidden_states.repeat(batch_size, 1, 1),
                            down_block_additional_residuals=down_res,
                            mid_block_additional_residual=mid_res,
                            added_cond_kwargs=unet_added_cond,
                            return_dict=False
                        )[0]

                    latents = pipeline.scheduler.step(noise_pred, t, latent_model_input).prev_sample
                gen_latent_feature = latents
                gen_output = vae.decode(gen_latent_feature / vae.config.scaling_factor).sample
                # Save to image
                pred_image = (gen_output.clamp(-1, 1) + 1) / 2  # scale to [0,1]
                for idx in range(pred_image.shape[0]):
                    img = pred_image[idx].detach().cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    pil_img.save(f"{args.pred_image_dir}/step{step}_t{i}_idx{idx}.png")
                    #Extract face embedding
                    emb = extract_face_embedding(np.array(pil_img)[:, :, ::-1])

                # Example backward and optimizer step (add your loss calculation here)
                # loss = ...
                # accelerator.backward(loss)
                # optimizer.step()
                # lr_scheduler.step()
                # optimizer.zero_grad()
                loss = F.mse_loss(emb, mean_face_embedding)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                if global_step >= args.max_train_steps:
                    break
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    if accelerator.is_main_process:
        vae_to_save = accelerator.unwrap_model(vae)
        ckpt_path = os.path.join(args.output_dir, "vae_finetuned.pth")
        torch.save(vae_to_save.state_dict(), ckpt_path)
        print(f"VAE checkpoint saved to {ckpt_path}")

            

        # for step, pose_batch in enumerate(pose_dataloader):
        #     pose_batch = pose_batch.to("cuda")

        #     # --- Feature extraction from UNet, conditioned on prompt and pose (via ControlNet) ---
        #     for i, t in enumerate(pipeline.scheduler.timesteps):
        #         features = {}
        #         def save_hook(name):
        #             def hook(module, input, output):
        #                 features[name] = output
        #             return hook

        #         # Register hook to a layer of interest (e.g., first down block)
        #         handles = []
        #         handles.append(unet.down_blocks[0].register_forward_hook(save_hook('down_block_0')))

        #         # Prepare latent input (random for demonstration)
        #         latent_shape = (pose_batch.shape[0], 4, 1024 // 8, 1024 // 8)
        #         dummy_latent = torch.randn(latent_shape, device="cuda")

        #         # Encode prompt
        #         encoder_hidden_states = prompt_hidden_states

        #         # Encode pose using controlnet
        #         with torch.no_grad():
        #             down_res, mid_res = controlnet(
        #                 dummy_latent,
        #                 t,
        #                 controlnet_cond=pose_batch,
        #                 encoder_hidden_states=encoder_hidden_states,
        #                 return_dict=False,
        #             )

        #             # UNet forward, conditioned on prompt and pose (via controlnet residuals)
        #             _ = unet(
        #                 dummy_latent,
        #                 t,
        #                 encoder_hidden_states,
        #                 down_block_additional_residuals=down_res,
        #                 mid_block_additional_residual=mid_res,
        #                 return_dict=False,
        #             )

        #         # Now 'features' dict contains the outputs of hooked layers
        #         print(f"Step {step}, Timestep {t}, Down Block 0 Feature Shape: {features['down_block_0'].shape}")

        #         # Remove hooks after use
        #         for h in handles:
        #             h.remove()


if __name__ == "__main__":
    args = parse_args()
    main(args)
