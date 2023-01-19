import itertools
from typing import Callable, Dict, List, Tuple

import keras_cv
import numpy as np
import tensorflow as tf
from imutils import paths
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

from src.constants import MAX_PROMPT_LENGTH, PADDING_TOKEN

POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
AUTO = tf.data.AUTOTUNE


class DatasetUtils:
    """
    DatasetUtils prepares a `tf.data.Dataset` object for DreamBooth training.
    It works in the following steps. First, it downloads images for instance
    and class (assuming they are compressed). Second, it optionally embeds the
    captions associated with the images with `TextEncoder`. Third, it builds
    `tf.data.Dataset` object of a pair of image and embeded text for instance
    and class separately. Finally, it zips the two `tf.data.Dataset` objects.
    """

    def __init__(
        self,
        instance_images_url: str,
        class_images_url: str,
        unique_id: str,
        class_category: str,
        train_text_encoder: bool,
        img_height: int = 512,
        img_width: int = 512,
        batch_size: int = 1,
    ):
        """
        Args:
            instance_images_url: URL of a compressed file which contains
                a set of instance images.
            class_images_url: URL of a compressed file which contains a
                set of class images.
            unique_id: unique identifier to represent a new concept/instance.
                for instance, the typically used unique_id is "sks" in DreamBooth.
            class_category: a class of concept which the unique_id belongs
                to. For instance, if unique_id represents a specific dog,
                class_category should be "dog".
            train_text_encoder: Boolean flag to denote if the text encoder
                is fine-tuned. If set to True, only tokenized text batches
                are passed to the trainer.
        """
        self.instance_images_url = instance_images_url
        self.class_images_url = class_images_url
        self.unique_id = unique_id
        self.class_category = class_category
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

        self.tokenizer = SimpleTokenizer()
        self.train_text_encoder = train_text_encoder
        if not self.train_text_encoder:
            self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

        self.augmenter = keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.CenterCrop(self.img_height, self.img_width),
                keras_cv.layers.RandomFlip(),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            ]
        )

    def _get_captions(
        self, num_instance_images: int, num_class_images: int
    ) -> Tuple[List, List]:
        """Prepares captions for instance and class images."""
        instance_caption = f"a photo of {self.unique_id} {self.class_category}"
        instance_captions = [instance_caption] * num_instance_images

        class_caption = f"a photo of {self.class_category}"
        class_captions = [class_caption] * num_class_images

        return instance_captions, class_captions

    def _tokenize_text(self, caption: str) -> np.ndarray:
        """Tokenizes a given caption."""
        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)

    def _tokenize_captions(
        self, instance_captions: List[str], class_captions: List[str]
    ) -> np.ndarray:
        """Tokenizes a batch of captions."""
        tokenized_texts = np.empty(
            (len(instance_captions) + len(class_captions), MAX_PROMPT_LENGTH)
        )
        for i, caption in enumerate(itertools.chain(instance_captions, class_captions)):
            tokenized_texts[i] = self._tokenize_text(caption)
        return tokenized_texts

    def _embed_captions(self, tokenized_texts: np.ndarray) -> np.ndarray:
        """Embeds captions with `TextEncoder`. This is done to save some memory."""
        # Ensure the computation takes place on a GPU.
        gpus = tf.config.list_logical_devices("GPU")
        with tf.device(gpus[0].name):
            embedded_text = self.text_encoder(
                [tf.convert_to_tensor(tokenized_texts), POS_IDS], training=False
            ).numpy()

        del self.text_encoder  # To ensure the GPU memory is freed.
        return embedded_text

    def _collate_instance_image_paths(
        self, instance_image_paths: List[str], class_image_paths: List[str]
    ) -> List:
        """Makes `instance_image_paths`'s length equal to the length of `class_image_paths`."""
        new_instance_image_paths = []
        for index in range(len(class_image_paths)):
            instance_image = instance_image_paths[index % len(instance_image_paths)]
            new_instance_image_paths.append(instance_image)

        return new_instance_image_paths

    def _download_images(self) -> Tuple[List, List]:
        """Downloads instance and class image archives from the URLs and
        un-archives them."""
        instance_images_root = tf.keras.utils.get_file(
            origin=self.instance_images_url,
            untar=True,
        )
        class_images_root = tf.keras.utils.get_file(
            origin=self.class_images_url,
            untar=True,
        )

        instance_image_paths = list(paths.list_images(instance_images_root))
        class_image_paths = list(paths.list_images(class_images_root))
        instance_image_paths = self._collate_instance_image_paths(
            instance_image_paths, class_image_paths
        )

        return instance_image_paths, class_image_paths

    def _process_image(
        self, image_path: tf.Tensor, text: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Reads an image file and scales it `text` can be either just tokens
        or embedded tokens."""
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        return image, text

    def _apply_augmentation(
        self, image_batch: tf.Tensor, text_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies data augmentation to a batch of images. `text_batch` can
        either be just tokens or embedded tokens."""
        return self.augmenter(image_batch), text_batch

    def _prepare_dict(self, instance_only=True) -> Callable:
        """
        Returns a function that returns a dictionary with an appropriate
        format for instance and class datasets.
        """

        def fn(image_batch, texts) -> Dict[str, tf.Tensor]:
            if instance_only:
                batch_dict = {
                    "instance_images": image_batch,
                    "instance_texts": texts,
                }
                return batch_dict
            else:
                batch_dict = {
                    "class_images": image_batch,
                    "class_texts": texts,
                }
                return batch_dict

        return fn

    def _assemble_dataset(
        self, image_paths: List[str], texts: np.ndarray, instance_only=True
    ) -> tf.data.Dataset:
        """Assembles `tf.data.Dataset` object from image paths and their corresponding
        captions. `texts` can either be tokens or embedded tokens."""
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, texts))
        dataset = dataset.map(self._process_image, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(5, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self._apply_augmentation, num_parallel_calls=AUTO)

        prepare_dict_fn = self._prepare_dict(instance_only=instance_only)
        dataset = dataset.map(prepare_dict_fn, num_parallel_calls=AUTO)
        return dataset

    def prepare_datasets(self) -> tf.data.Dataset:
        """Prepares dataset for DreamBooth training.

        1. Download the instance and class images (archives) and un-archive them.
        2. Prepare the instance and class image paths.
        3. Prepare the captions.
        4. Tokenize the captions.
        5. If the text encoder is NOT fine-tuned then embed the tokenized captions.
        6. Assemble the datasets.
        """
        print("Downloading instance and class images...")
        instance_image_paths, class_image_paths = self._download_images()

        # Prepare captions.
        instance_captions, class_captions = self._get_captions(
            len(instance_image_paths), len(class_image_paths)
        )
        # Tokenize the captions.
        text_batch = self._tokenize_captions(instance_captions, class_captions)

        # `text_batch` can either be embedded captions or tokenized captions.
        if not self.train_text_encoder:
            print("Embedding captions via TextEncoder...")
            text_batch = self._embed_captions(text_batch)

        print("Assembling instance and class datasets...")
        instance_dataset = self._assemble_dataset(
            instance_image_paths,
            text_batch[: len(instance_image_paths)],
        )
        class_dataset = self._assemble_dataset(
            class_image_paths,
            text_batch[len(instance_image_paths) :],
            instance_only=False,
        )

        train_dataset = tf.data.Dataset.zip((instance_dataset, class_dataset))
        return train_dataset.prefetch(AUTO)
