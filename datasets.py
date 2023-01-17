import logging
import itertools

import numpy as np
import tensorflow as tf
import keras_cv
from imutils import paths
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from constants import PADDING_TOKEN, MAX_PROMPT_LENGTH, RESOLUTION

AUTO = tf.data.AUTOTUNE

augmenter = keras_cv.layers.Augmenter(
    layers=[
        keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

def process_text(caption):
    """tokenize the caption"""

    tokenizer = SimpleTokenizer()

    tokens = tokenizer.encode(caption)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    return np.array(tokens)

def get_captions(
    num_instance_images, num_class_images,
    unique_id, class_category
):
    """prepare captions for instance and class images"""

    instance_caption = f"a photo of {unique_id} {class_category}"
    instance_captions = [instance_caption] * num_instance_images

    class_caption = f"a photo of {class_category}"
    class_captions = [class_caption] * num_class_images

    return instance_captions, class_captions


def get_embedded_caption(
    num_instance_images, num_class_images,
    unique_id, class_category
):
    """get embedded text for each caption with TextEncoder"""

    instance_captions, class_captions = get_captions(
        num_instance_images, num_class_images,
        unique_id, class_category
    )

    # Collate the tokenized captions into an array.
    tokenized_texts = np.empty(
        (num_instance_images + num_class_images, MAX_PROMPT_LENGTH)
    )
    for i, caption in enumerate(itertools.chain(instance_captions, class_captions)):
        tokenized_texts[i] = process_text(caption)

    # We also pre-compute the text embeddings to save some memory during training.
    POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
    text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

    embedded_text = text_encoder(
        [tf.convert_to_tensor(tokenized_texts), POS_IDS], training=False
    ).numpy()

    return embedded_text

def collate_instance_image_paths(instance_image_paths, class_image_paths):
    """make instance_image_paths's length equal to the length of class_image_paths"""

    new_instance_image_paths = []
    for index in range(len(class_image_paths)):
        instance_image = instance_image_paths[index % len(instance_image_paths)]
        new_instance_image_paths.append(instance_image)

    return new_instance_image_paths    

def download_images(
    instance_images_url,
    class_images_url
):
    """download instance and class images(compressed) from the urls"""

    instance_images_root = tf.keras.utils.get_file(
        origin=instance_images_url, untar=True,
    )
    class_images_root = tf.keras.utils.get_file(
        origin=class_images_url, untar=True,
    )

    instance_image_paths = list(paths.list_images(instance_images_root))
    class_image_paths = list(paths.list_images(class_images_root))
    instance_image_paths = collate_instance_image_paths(instance_image_paths, class_image_paths)

    return instance_image_paths, class_image_paths


def process_image(image_path, tokenized_text):
    """read image file and scale"""

    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 3)
    image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
    return image, tokenized_text


def apply_augmentation(image_batch, embedded_tokens):
    """apply data augmentation to a batch of images"""

    return augmenter(image_batch), embedded_tokens


def prepare_dict(instance_only=True):
    """
    get a function that returns a dictionary with an appropriate
    format for instance and class datasets
    """

    def fn(image_batch, embedded_tokens):
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


def assemble_dataset(image_paths, embedded_texts, instance_only=True, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, embedded_texts))
    dataset = dataset.map(process_image, num_parallel_calls=AUTO)
    dataset = dataset.shuffle(5, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)

    prepare_dict_fn = prepare_dict(instance_only=instance_only)
    dataset = dataset.map(prepare_dict_fn, num_parallel_calls=AUTO)
    return dataset


def prepare_datasets(
    instance_images_url, class_images_url,
    unique_id, class_category
):
    """get tf.data of zipped instance and class datasets"""

    print("downloading instance and class images")
    instance_image_paths, class_image_paths = download_images(
        instance_images_url, class_images_url
    )

    print("preparing embeded caption via TextEncoder")
    embedded_text = get_embedded_caption(
        len(instance_image_paths), len(class_image_paths),
        unique_id, class_category
    )

    print("assembling instance and class dataset")
    instance_dataset = assemble_dataset(
        instance_image_paths,
        embedded_text[: len(instance_image_paths)],
    )
    class_dataset = assemble_dataset(
        class_image_paths, 
        embedded_text[len(instance_image_paths) :], 
        instance_only=False
    )

    train_dataset = tf.data.Dataset.zip(
        (instance_dataset, class_dataset)
    )
    return train_dataset
    