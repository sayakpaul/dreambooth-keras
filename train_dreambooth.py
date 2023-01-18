import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import math

import tensorflow as tf
import wandb
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from tensorflow.keras import mixed_precision

from src import utils
from src.constants import MAX_PROMPT_LENGTH
from src.datasets import DatasetUtils
from src.dreambooth_trainer import DreamBoothTrainer


# These hyperparameters come from this tutorial by Hugging Face:
# https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
def get_optimizer(lr=5e-6, beta_1=0.9, beta_2=0.999, weight_decay=(1e-2,), epsilon=1e-08):
    """Instantiates the AdamW optimizer."""

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )

    return optimizer


def prepare_trainer(img_resolution: int, use_mp: bool):
    """Instantiates and compiles `DreamBoothTrainer` for training."""
    image_encoder = ImageEncoder(img_resolution, img_resolution)

    dreambooth_trainer = DreamBoothTrainer(
        diffusion_model=DiffusionModel(img_resolution, img_resolution, MAX_PROMPT_LENGTH),
        # Remove the top layer from the encoder, which cuts off
        # the variance and only returns the mean.
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        ),
        noise_scheduler=NoiseScheduler(),
        use_mixed_precision=use_mp,
    )

    optimizer = get_optimizer()
    dreambooth_trainer.compile(optimizer=optimizer, loss="mse")
    print("DreamBooth trainer initialized and compiled.")

    return dreambooth_trainer


def train(dreambooth_trainer, train_dataset, ckpt_path_prefix, max_train_steps):
    """Performs DreamBooth training `DreamBoothTrainer` with the given `train_dataset`."""
    num_update_steps_per_epoch = train_dataset.cardinality()
    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    print(f"Training for {epochs} epochs.")

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path_prefix,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    dreambooth_trainer.fit(train_dataset, epochs=epochs, callbacks=[ckpt_callback])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to perform DreamBooth training using Stable Diffusion."
    )
    # Dataset related.
    parser.add_argument(
        "--instance_images_url",
        default="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz",
        type=str,
    )
    parser.add_argument(
        "--class_images_url",
        default="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz",
        type=str,
    )
    parser.add_argument("--unique_id", default="sks", type=str)
    parser.add_argument("--class_category", default="dog", type=str)
    parser.add_argument("--img_resolution", default=512, type=int)
    # Optimization hyperparameters.
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    # Training hyperparameters.
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_train_steps", default=800, type=int)
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="If fine-tune the text-encoder too.",
    )
    parser.add_argument(
        "--mp", action="store_true", help="Whether to use mixed-precision."
    )
    # Misc.
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to use mixed-precision."
    )
    parser.add_argument(
        "--validation_prompt",
        default=None,
        type=str,
        help="Prompt to generate samples for validation purposes.",
    )

    return parser.parse_args()


def run(args):
    if args.mp:
        print("Enabling mixed-precision...")
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    print("Initializing dataset...")
    data_util = DatasetUtils(
        instance_images_url=args.instance_images_url,
        class_images_url=args.class_images_url,
        unique_id=args.unique_id,
        class_category=args.class_category,
        train_text_encoder=args.train_text_encoder,
        batch_size=args.batch_size,
    )
    train_dataset = data_util.prepare_datasets()

    print("Initializing trainer...")
    ckpt_path_prefix = "dreambooth"
    dreambooth_trainer = prepare_trainer(args.img_resolution, args.mp)
    train(dreambooth_trainer, train_dataset, ckpt_path_prefix, args.max_train_steps)

    if args.log_wandb:
        print("Logging artifacts...")
        wandb.init(project="dreambooth-keras", config=vars(args))
        if args.validation_prompt is None:
            args.validation_prompt = (
                f"A photo of {args.unique_id} {args.class_category} in a bucket"
            )

        ckpt_paths = [dreambooth_trainer.diffusion_model_path]
        if args.train_text_encoder:
            ckpt_paths.extend(dreambooth_trainer.text_encoder_model_path)
        utils.log_images(
            ckpt_paths,
            img_heigth=args.img_resolution,
            img_width=args.img_resolution,
            prompt=args.validation_prompt,
        )
        utils.save_ckpts(ckpt_paths)


if __name__ == "__main__":
    args = parse_args()
    run(args)
