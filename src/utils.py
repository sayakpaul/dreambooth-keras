from glob import glob

import keras_cv
from typing import List

import PIL
from tqdm.auto import tqdm

import tensorflow as tf

import wandb
from wandb.keras import WandbModelCheckpoint


def log_images(ckpt_paths, img_heigth, img_width, prompts, num_imgs_to_gen=5):
    """Logs generated images to WandB for qualitative validation."""
    print("Performing inference for logging generated images...")
    print(f"Number of images to generate: {num_imgs_to_gen}")

    sd_model = keras_cv.models.StableDiffusion(img_height=img_heigth, img_width=img_width)
    sd_model.diffusion_model.load_weights(ckpt_paths[0])
    if len(ckpt_paths) > 1:
        sd_model.text_encoder.load_weights(ckpt_paths[1])

    for prompt in tqdm(prompts):
        images_dreamboothed = sd_model.text_to_image(prompt, batch_size=num_imgs_to_gen)
        wandb.log(
            {
                f"validation/Prompt: {prompt}": [
                    wandb.Image(PIL.Image.fromarray(image), caption=f"{i}: {prompt}")
                    for i, image in enumerate(images_dreamboothed)
                ]
            }
        )


class QualitativeValidationCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        img_heigth: int,
        img_width: int,
        prompts: List[str],
        num_imgs_to_gen: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_heigth = img_heigth
        self.img_width = img_width
        self.prompts = prompts
        self.num_imgs_to_gen = num_imgs_to_gen
        self.sd_model = keras_cv.models.StableDiffusion(
            img_height=self.img_heigth, img_width=self.img_width
        )
        self.wandb_table = wandb.Table(columns=["epoch", "prompt", "images"])

    def on_epoch_end(self, epoch, logs=None):
        print(f"Performing inference for logging generated images for epoch {epoch}...")
        print(f"Number of images to generate: {self.num_imgs_to_gen}")

        # load weights to stable diffusion model
        self.sd_model.diffusion_model.set_weights(
            self.model.diffusion_model.get_weights()
        )
        if hasattr(self.model, "text_encoder"):
            self.sd_model.text_encoder.set_weights(
                self.model.text_encoder.get_weights()
            )

        for prompt in self.prompts:
            images_dreamboothed = self.sd_model.text_to_image(
                prompt, batch_size=self.num_imgs_to_gen
            )
            images_dreamboothed = [
                wandb.Image(PIL.Image.fromarray(image), caption=f"{i}: {prompt}")
                for i, image in enumerate(images_dreamboothed)
            ]
            self.wandb_table.add_data(epoch, prompt, images_dreamboothed)

    def on_train_end(self, logs=None):
        wandb.log({"validation-table": self.wandb_table})


class DreamBoothCheckpointCallback(WandbModelCheckpoint):
    def __init__(
        self, filepath, save_weights_only: bool = False, *args, **kwargs
    ) -> None:
        super(DreamBoothCheckpointCallback.__bases__[0], self).__init__(
            filepath, save_weights_only=save_weights_only, *args, **kwargs
        )
        self.save_weights_only = save_weights_only
        # User-friendly warning when trying to save the best model.
        if self.save_best_only:
            self._check_filepath()
        self._is_old_tf_keras_version = None

    def _log_ckpt_as_artifact(self, filepath: str, aliases) -> None:
        if wandb.run is not None:
            model_artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
            for file in glob(f"{filepath}*.h5"):
                model_artifact.add_file(file)
            wandb.log_artifact(model_artifact, aliases=aliases or [])
