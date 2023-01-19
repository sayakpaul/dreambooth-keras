# Implementation of DreamBooth using KerasCV and TensorFlow

This repository provides an implementation of [DreamBooth](https://arxiv.org/abs/2208.12242) using KerasCV and TensorFlow. The implementation is heavily referred from Hugging Face's `diffusers` [example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth).

DreamBooth is a way of quickly teaching (fine-tuning) Stable Diffusion about new visual concepts. For more details, refer to [this document](https://dreambooth.github.io/).

## Steps to perform DreamBooth training using the codebase

1. Install the pre-requisites: `pip install -r requirements.txt`.

2. You first need to choose a class to which a unique identifier is appended. This repository codebase was tested using `sks` as the unique idenitifer and `dog` as the class.

    Then two types of prompts are generated: 

    (a) **instance prompt**: f"a photo of {self.unique_id} {self.class_category}"
    (b) **class prompt**: f"a photo of {self.class_category}"

3. **Instance images**
    
    Get a few images (3 - 10) that are representative of the concept the model is going to be fine-tuned with. These images would be associated with the `instance_prompt`. These images are referred to as the `instance_images` from the codebase. Archive these images and host them somewhere online such that the archive can be downloaded using `tf.keras.utils.get_file()` function internally.

4. **Class images**
    
    DreamBooth uses prior-preservation loss to regularize training. Long story cut short,
prior-preservation loss helps the model to slowly adapt to the new concept under consideration from any prior knowledge it may have had about the concept. To use prior-preservation loss, we need the class prompt as shown above. The class prompt is used to generate a pre-defined number of images which are used for computing the final loss used for DreamBooth training. 

    As per [this resource](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth), 200 - 300 images generated using the class prompt work well for most cases. 

    So, after you have decided `instance_prompt` and `class_prompt`, use [this Colab Notebook](https://colab.research.google.com/gist/sayakpaul/6b5de345d29cf5860f84b6d04d958692/generate_class_priors.ipynb) to generate some images that would be used for training with the prior-preservation loss. Then archive the generated images as a single archive and host it online such that it can be downloaded using using `tf.keras.utils.get_file()` function internally. In the codebase, we simply refer to these images as `class_images`.
    
> For people to easily test this codebase, we hosted the instance and class images [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/tree/main). 

5. Launch training! There are a number of hyperparameters you can play around with. Refer to the `train_dreambooth.py` script to know more about them. Here's a command that launches training with mixed-precision and other default values:

```bash=
python train_dreambooth.py --mp
```

You can also fine-tune the text encoder by specifying the `--train_text_encoder` option. 

Additionally, the script supports integration with [Weights and Biases (`wandb`)](https://wandb.ai/). if you specify `--log_wandb`, then it will perform inference with the DreamBoothed model parameters and log the generated images to `wandb` alongside the model parameters as artifacts. [Here's](https://wandb.ai/sayakpaul/dreambooth-keras/runs/este2e4c) an example `wandb` run where you can find the generated images as well as the [model parameters](https://wandb.ai/sayakpaul/dreambooth-keras/artifacts/model/run_este2e4c_model/v0/files). 

## Inference

TBA

## Results

## Acknowledgements

* Thanks to Hugging Face for providing the original example. It's very readable and easy to understand.
* Thanks to the ML Developer Programs' team at Google for providing GCP credits.