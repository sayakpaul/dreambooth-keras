import keras_cv
import PIL
import wandb


def log_images(ckpt_paths, img_heigth, img_width, prompt, num_imgs_to_gen=5):
    """Logs generated images to WandB for qualitative validation."""
    print("Performing inference for logging generated images...")
    print(f"Number of images to generate: {num_imgs_to_gen} to prompt: {prompt}")

    sd_model = keras_cv.models.StableDiffusion(img_height=img_heigth, img_width=img_width)
    sd_model.diffusion_model.load_weights(ckpt_paths[0])

    images_dreamboothed = sd_model.text_to_image(prompt, batch_size=num_imgs_to_gen)
    wandb.log(
        {
            "validation": [
                wandb.Image(PIL.Image.fromarray(image), caption=f"{i}: {prompt}")
                for i, image in enumerate(images_dreamboothed)
            ]
        }
    )


def save_ckpts(ckpt_paths):
    """Saves model checkpoints as WandB artifacts."""
    print(f"Serializing the following checkpoints: \n {ckpt_paths}")
    model_artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
    for ckpt_path in ckpt_paths:
        model_artifact.add_file(ckpt_path)
    wandb.log_artifact(model_artifact, aliases=[f"run_{wandb.run.id}"])
