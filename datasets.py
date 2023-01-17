from typing import List, Dict, Tuple, Callable

import logging
import itertools

import numpy as np
import tensorflow as tf
import keras_cv
from imutils import paths
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

from constants import PADDING_TOKEN, MAX_PROMPT_LENGTH

POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
AUTO = tf.data.AUTOTUNE

class DatasetUtils:
    def __init__(
        self,
        instance_images_url: str,
        class_images_url: str,
        unique_id: str, 
        class_category: str,
        img_height: int = 512,
        img_width: int = 512,
        batch_size: int = 1
    ):

        self.instance_images_url = instance_images_url
        self.class_images_url = class_images_url
        self.unique_id = unique_id
        self.class_category = class_category
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

        self.augmenter = keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.CenterCrop(self.img_height, self.img_width),
                keras_cv.layers.RandomFlip(),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            ]
        )

    def _process_text(self, caption) -> np.ndarray:
        """Tokenizes a given caption."""

        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)

    def _get_captions(
        self,
        num_instance_images, 
        num_class_images,
    ) -> Tuple[List, List]:
        """Prepares captions for instance and class images."""

        instance_caption = f"a photo of {self.unique_id} {self.class_category}"
        instance_captions = [instance_caption] * num_instance_images

        class_caption = f"a photo of {self.class_category}"
        class_captions = [class_caption] * num_class_images

        return instance_captions, class_captions

    def _get_embedded_caption(
        self,
        num_instance_images, 
        num_class_images
    ) -> np.ndarray:
        """Embeds captions with `TextEncoder`."""

        instance_captions, class_captions = self._get_captions(
            num_instance_images, num_class_images,
        )

        # Collate the tokenized captions into an array.
        tokenized_texts = np.empty(
            (num_instance_images + num_class_images, MAX_PROMPT_LENGTH)
        )
        for i, caption in enumerate(itertools.chain(instance_captions, class_captions)):
            tokenized_texts[i] = self._process_text(caption)

        gpus = tf.config.list_logical_devices("GPU")

        # Ensure the computation takes place on a GPU.
        with tf.device(gpus[0].name):
            embedded_text = self.text_encoder(
                [tf.convert_to_tensor(tokenized_texts), POS_IDS], training=False
            ).numpy()

        return embedded_text

    def _collate_instance_image_paths(
        self,
        instance_image_paths, 
        class_image_paths
    ) -> List:
        """Makes `instance_image_paths`'s length equal to the length of `class_image_paths`."""

        new_instance_image_paths = []
        for index in range(len(class_image_paths)):
            instance_image = instance_image_paths[index % len(instance_image_paths)]
            new_instance_image_paths.append(instance_image)

        return new_instance_image_paths    

    def _download_images(self) -> Tuple[List, List]:
        """Downloads instance and class image archives from the URLs and un-archives them."""

        instance_images_root = tf.keras.utils.get_file(
            origin=self.instance_images_url, untar=True,
        )
        class_images_root = tf.keras.utils.get_file(
            origin=self.class_images_url, untar=True,
        )

        instance_image_paths = list(paths.list_images(instance_images_root))
        class_image_paths = list(paths.list_images(class_images_root))
        instance_image_paths = self._collate_instance_image_paths(
            instance_image_paths, class_image_paths
        )

        return instance_image_paths, class_image_paths

    def _process_image(
        self,
        image_path, 
        tokenized_text
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Reads an image file and scales it."""

        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        return image, tokenized_text


    def _apply_augmentation(
        self, 
        image_batch, 
        embedded_tokens
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies data augmentation to a batch of images."""

        return self.augmenter(image_batch), embedded_tokens


    def _prepare_dict(self, instance_only=True) -> Callable:
        """
        Returns a function that returns a dictionary with an appropriate
        format for instance and class datasets.
        """

        def fn(image_batch, embedded_tokens) -> Dict[str, tf.Tensor]:
            if instance_only:
                batch_dict = {
                    "instance_images": image_batch,
                    "instance_embedded_texts": embedded_tokens,
                }
                return batch_dict
            else:
                batch_dict = {
                    "class_images": image_batch,
                    "class_embedded_texts": embedded_tokens,
                }
                return batch_dict

        return fn


    def _assemble_dataset(
        self,
        image_paths, 
        embedded_texts, 
        instance_only=True
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, embedded_texts))
        dataset = dataset.map(self._process_image, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(5, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self._apply_augmentation, num_parallel_calls=AUTO)

        prepare_dict_fn = self._prepare_dict(instance_only=instance_only)
        dataset = dataset.map(prepare_dict_fn, num_parallel_calls=AUTO)
        return dataset


    def prepare_datasets(self) -> tf.data.Dataset:
        """Prepares dataset for DreamBooth training."""

        print("downloading instance and class images")
        instance_image_paths, class_image_paths = self._download_images()

        print("preparing embeded caption via TextEncoder")
        embedded_text = self._get_embedded_caption(
            len(instance_image_paths), len(class_image_paths),
        )

        print("assembling instance and class dataset")
        instance_dataset = self._assemble_dataset(
            instance_image_paths,
            embedded_text[: len(instance_image_paths)],
        )
        class_dataset = self._assemble_dataset(
            class_image_paths, 
            embedded_text[len(instance_image_paths) :], 
            instance_only=False
        )

        train_dataset = tf.data.Dataset.zip(
            (instance_dataset, class_dataset)
        )
        return train_dataset.prefetch(AUTO)
