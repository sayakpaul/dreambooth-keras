import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

import glob
import os

import keras_cv
import numpy as np
import PIL
import wandb
from tqdm import tqdm

def generate_report(
    sd_model, 
    weights_dict,
    caption,
    num_image_gen,
    num_steps,
    unconditional_guidance_scales,
    wandb_project
):
    "generate images and report the results to W&B"
    for key in list(weights_dict.keys()):
        print(f"Generating images for model({key}).")
        wandb.init(project=wandb_project, name=key)

        unet_params_path = weights_dict[key]['unet']
        sd_model.diffusion_model.load_weights(unet_params_path)
        
        if 'text_encoder' in weights_dict[key]:
            text_encoder_params_path = weights_dict[key]['text_encoder']
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
    """find weights per model name"""

    weights_dict = {}

    for file in glob.glob(f"{base_root_dir}/*.h5"):
        if "@True" in file:
            rindex = file.rindex("@True")
            key = file[:rindex+len("@True")]
        else:
            rindex = file.rindex("@False")
            key = file[:rindex+len("@False")]            
        
        if key not in weights_dict:
            weights_dict[key] = {}

        if "text_encoder" in file[rindex:]:
            weights_dict[key]["text_encoder"] = file
        else:
            weights_dict[key]["unet"] = file

    return weights_dict
    
def run(args):
    """find weights, generate images based on them"""

    # Initialize the SD model.
    img_height = img_width = 512
    sd_model = keras_cv.models.StableDiffusion(
        img_width=img_width, img_height=img_height, jit_compile=True
    )    
    
    # Find weights per model
    weights_dict = find_weights(args.base_root_dir)

    # Run image generations
    generate_report(
        sd_model, 
        weights_dict,
        args.caption,
        args.num_image_gen,
        args.num_steps,
        args.ugs,
        args.wandb_project_id)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to perform image generating experimentations."
    )

    parser.add_argument("--base_root_dir", type=str,
                        default=".",
                        help="base directory to search for weight files")
    parser.add_argument("--caption", type=str, 
                        default="A photo of sks person without mustache, handsome, ultra realistic, 4k, 8k",
                        help="prompt to use to generate images")
    parser.add_argument("--num_image_gen", type=int, 
                        default=16,
                        help="number of images to generate per model")
    parser.add_argument("--num_steps", nargs="+", type=int, 
                        default=[75, 100, 150],
                        help="list of num_steps")
    parser.add_argument("--ugs", nargs="+", type=int, 
                        default=[15, 30],
                        help="list of unconditional guidance scale")

    parser.add_argument("--wandb_project_id", type=str, 
                        default="dreambooth-generate-me",
                        help="W&B project id to log")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
