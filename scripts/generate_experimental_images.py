"""
    Use this script to generate and report a batch of images to W&B 
    with a single prompt through multiple weights of Stable Diffusion.
    This is particularly useful when you have multiple fine-tuned weight 
    files. It works for the both cases: (a) diffusion model only and
    (b) text encoder + diffusion model

    Usage:

    # Find weight files(.h5) under "." location.
    # Generate 4 images with a nested loop over
    # num_steps x ugs combinations.
    $ python generate_experimental_images.py \
        --base_root_dir "." \   
        --caption "A photo of sks dog in a bucket" \ 
        --num_image_gen 4 \
        --num_steps 75 100 150 \
        --ugs 15 30 \
        --wandb_project_id "my-wandb-project"

    Depending on the unique identifier and class you used, you'd need
    to change the caption accordingly. In the above case, the unique identifier
    is "sks" and the class is "dog".

    If the fine-tuned weights are stored as artifacts in WandB, then you can
    use this script: https://gist.github.com/sayakpaul/0d83d7fd7c3939ce2ddc2292b6d4f173
"""

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

import argparse
import glob

import keras_cv
import PIL
import wandb


def generate_report(
    sd_model,
    weights_dict,
    caption,
    num_image_gen,
    num_steps,
    unconditional_guidance_scales,
    wandb_project,
):
    "Generates images and report the results to WandB."
    for key in list(weights_dict.keys()):
        print(f"Generating images for model({key}).")
        wandb.init(project=wandb_project, name=key)

        unet_params_path = weights_dict[key]["unet"]
        sd_model.diffusion_model.load_weights(unet_params_path)

        if "text_encoder" in weights_dict[key]:
            text_encoder_params_path = weights_dict[key]["text_encoder"]
            sd_model.text_encoder.load_weights(text_encoder_params_path)

        for steps in num_steps:
            for ugs in unconditional_guidance_scales:
                images = sd_model.text_to_image(
                    caption,
                    batch_size=num_image_gen,
                    num_steps=steps,
                    unconditional_guidance_scale=ugs,
                )

                wandb.log(
                    {
                        f"num_steps@{steps}-ugs@{ugs}": [
                            wandb.Image(
                                PIL.Image.fromarray(image), caption=f"{i}: {caption}"
                            )
                            for i, image in enumerate(images)
                        ]
                    }
                )

        wandb.finish()


def find_weights(base_root_dir):
    """Finds weights per model name."""
    weights_dict = {}

    for file in glob.glob(f"{base_root_dir}/*.h5"):
        if "@True" in file:
            rindex = file.rindex("@True")
            key = file[: rindex + len("@True")]
        else:
            rindex = file.rindex("@False")
            key = file[: rindex + len("@False")]

        if key not in weights_dict:
            weights_dict[key] = {}

        if "text_encoder" in file[rindex:]:
            weights_dict[key]["text_encoder"] = file
        else:
            weights_dict[key]["unet"] = file

    return weights_dict


def run(args):
    """Finds weights, generate images based on them"""
    # Initialize the SD model.
    img_height = img_width = 512
    sd_model = keras_cv.models.StableDiffusion(
        img_width=img_width, img_height=img_height, jit_compile=True
    )

    # Find weights per model.
    weights_dict = find_weights(args.base_root_dir)

    # Run image generations.
    generate_report(
        sd_model,
        weights_dict,
        args.caption,
        args.num_image_gen,
        args.num_steps,
        args.ugs,
        args.wandb_project_id,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to perform image generating experimentations."
    )

    parser.add_argument(
        "--base_root_dir",
        type=str,
        default=".",
        help="base directory to search for weight files",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="A photo of sks person without mustache, handsome, ultra realistic, 4k, 8k",
        help="prompt to use to generate images",
    )
    parser.add_argument(
        "--num_image_gen",
        type=int,
        default=16,
        help="number of images to generate per model",
    )
    parser.add_argument(
        "--num_steps",
        nargs="+",
        type=int,
        default=[75, 100, 150],
        help="list of num_steps",
    )
    parser.add_argument(
        "--ugs",
        nargs="+",
        type=int,
        default=[15, 30],
        help="list of unconditional guidance scale",
    )

    parser.add_argument(
        "--wandb_project_id",
        type=str,
        default="dreambooth-generate-dog",
        help="W&B project id to log",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
