# Implementation of DreamBooth using KerasCV and TensorFlow

This repository provides an implementation of [DreamBooth](https://arxiv.org/abs/2208.12242) using KerasCV and TensorFlow. The implementation is heavily referred from Hugging Face's `diffusers` [example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth).

DreamBooth is a way of quickly teaching (fine-tuning) Stable Diffusion about new visual concepts. For more details, refer to [this document](https://dreambooth.github.io/).

**The code provided in this repository is for research purposes only**. Please check out [this section](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion#uses) to know more about the potential use cases and limitations.

By loading this model you accept the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE.

<div align="center">
<img src="https://i.imgur.com/gYlgLPm.png"/>
</div>

If you're just looking for the accompanying resources of this repository, here are the links:

* [Inference Colab Notebook](https://colab.research.google.com/github/sayakpaul/dreambooth-keras/blob/main/notebooks/inference_dreambooth.ipynb)
* [Blog post on keras.io] (upcoming)
* [Fine-tuned model weights](https://huggingface.co/chansung/dreambooth-dog)

### Table of contents

* [Performing DreamBooth training with the codebase](#steps-to-perform-dreambooth-training-using-the-codebase)
* [Running inference](#inference)
* [Results](#results)
* [Using in Diffusers 🧨](#using-in-diffusers-)
* [Notes](#notes-on-preparing-data-for-dreambooth-training-of-faces)
* [Acknowledgements](#acknowledgements)

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

    So, after you have decided `instance_prompt` and `class_prompt`, use [this Colab Notebook](https://colab.research.google.com/github/sayakpaul/dreambooth-keras/blob/main/notebooks/generate_class_priors.ipynb) to generate some images that would be used for training with the prior-preservation loss. Then archive the generated images as a single archive and host it online such that it can be downloaded using using `tf.keras.utils.get_file()` function internally. In the codebase, we simply refer to these images as `class_images`.
    
> It's possible to conduct DreamBooth training WITHOUT using a prior preservation loss. This repository always uses it. For people to easily test this codebase, we hosted the instance and class images [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/tree/main). 

5. Launch training! There are a number of hyperparameters you can play around with. Refer to the `train_dreambooth.py` script to know more about them. Here's a command that launches training with mixed-precision and other default values:

    ```bash
    python train_dreambooth.py --mp
    ```

    You can also fine-tune the text encoder by specifying the `--train_text_encoder` option. 

    Additionally, the script supports integration with [Weights and Biases (`wandb`)](https://wandb.ai/). If you specify `--log_wandb`, then it will perform inference with the DreamBoothed model parameters and log the generated images to `wandb` alongside the model parameters as artifacts. [Here's](https://wandb.ai/sayakpaul/dreambooth-keras/runs/este2e4c) an example `wandb` run where you can find the generated images as well as the [model parameters](https://wandb.ai/sayakpaul/dreambooth-keras/artifacts/model/run_este2e4c_model/v0/files). 

## Inference

* [Colab Notebook](https://colab.research.google.com/github/sayakpaul/dreambooth-keras/blob/main/notebooks/inference_dreambooth.ipynb)
* [Script for launching bulk experiments](https://github.com/sayakpaul/dreambooth-keras/blob/main/scripts/generate_experimental_images.py)

## Results

We have tested our implementation in two different methods: (a) fine-tuning the diffusion model (the UNet) only, (b) fine-tuning the diffusion model along with the text encoder. The experiments were conducted over a wide range of hyperparameters for `learning rate` and `training steps` for during training and for `number of steps` and `unconditional guidance scale` (ugs) during inference. But only the most salient results (from our perspective) are included here. If you are curious about how different hyperparameters affect the generated image quality, find the link to the full reports in each section.

__Note that our experiments were guided by [this blog post from Hugging Face](https://huggingface.co/blog/dreambooth).__

### (a) Fine-tuning diffusion model

Here are a selected few results from various experiments we conducted. Our experimental logs for this setting are available [here](https://wandb.ai/sayakpaul/dreambooth-keras). More visualization images (generated with the checkpoints from these experiments) are available [here](https://wandb.ai/sayakpaul/experimentation_images). 


<div align="center">
<table>
  <tr>
    <th>Images</th>
    <th>Steps</th>
    <th>UGS</th>
    <th>Setting</th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/UUSfrwW.png"/></td>
    <td>50</td>
    <td>30</td>
    <td>LR: 1e-6 Training steps: 800 <a href="https://huggingface.co/sayakpaul/dreambooth-keras-dogs-unet/resolve/main/lr_1e-6_steps_800_unet.h5">(Weights)</a></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/Ewt0BhG.png"/></td>
    <td>25</td>
    <td>15</td>
    <td>LR: 1e-6 Training steps: 1000 <a href="https://huggingface.co/sayakpaul/dreambooth-keras-dogs-unet/resolve/main/lr_1e-6_steps_1000.h5">(Weights)</a></td>
  </tr>  
  <tr>
    <td><img src="https://i.imgur.com/Dn0uGZa.png"/></td>
    <td>75</td>
    <td>15</td>
    <td>LR: 3e-6 Training steps: 1200 <a href="https://huggingface.co/sayakpaul/dreambooth-keras-dogs-unet/resolve/main/lr_3e-6_steps_1200_unet.h5">(Weights)</a></td>
  </tr>
</table>
<sub><b>Caption</b>: "A photo of sks dog in a bucket" </sub> 
</div>

### (b) Fine-tuning text encoder + diffusion model

<div align="center">
<table>
  <tr>
    <th>Images</th>
    <th>Steps</th>
    <th>ugs</th>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/BNVtwDB/dog.png"/></td>
    <td>75</td>
    <td>15</td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/zWMzxq2/dog-2.png"/></td>
    <td>75</td>
    <td>30</td>
  </tr>  
</table>
<sub>"<b>Caption</b>: A photo of sks dog in a bucket" </sub> 

<sub> w/ learning rate=9e-06, max train steps=200 (<a href="https://huggingface.co/chansung/dreambooth-dog">weights</a> | <a href="https://wandb.ai/chansung18/dreambooth-keras-generating-images?workspace=user-chansung18">reports</a>)</sub>
</div><br>


<div align="center">
<table>
  <tr>
    <th>Images</th>
    <th>Steps</th>
    <th>ugs</th>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/XYz3s5N/chansung.png"/></td>
    <td>150</td>
    <td>15</td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/mFMZG04/chansung-2.png"/></td>
    <td>75</td>
    <td>30</td>
  </tr>  
</table>
<sub>"<b>Caption</b>: A photo of sks person without mustache, handsome, ultra realistic, 4k, 8k"</sub> 

<sub> w/ learning rate=9e-06, max train steps=200 (<a href="https://huggingface.co/datasets/chansung/me">datasets</a> | <a href="https://wandb.ai/chansung18/dreambooth-generate-me?workspace=user-chansung18">reports</a>)</sub>
</div><br>

## Using in Diffusers 🧨

The [`diffusers` library](https://github.com/huggingface/diffusers/) provides state-of-the-art tooling for experimenting with
different Diffusion models, including Stable Diffusion. It includes 
different optimization techniques that can be leveraged to perform efficient inference
with `diffusers` when using large Stable Diffusion checkpoints. One particularly 
advantageous feature `diffusers` has is its support for [different schedulers](https://huggingface.co/docs/diffusers/using-diffusers/schedulers) that can
be configured during runtime and can be integrated into any compatible Diffusion model.

Once you have obtained the DreamBooth fine-tuned checkpoints using this codebase, you can actually
export those into a handy [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) and use it from the `diffusers` library directly. 

Consider this repository: [chansung/dreambooth-dog](https://huggingface.co/chansung/dreambooth-dog). You can use the
checkpoints of this repository in a `StableDiffusionPipeline` after running some small steps:

```py
from diffusers import StableDiffusionPipeline

model_ckpt = "sayakpaul/text-unet-dogs-kerascv_sd_diffusers_pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt)
pipeline.to("cuda")

unique_id = "sks"
class_label = "dog"
prompt = f"A photo of {unique_id} {class_label} in a bucket"
image = pipeline(prompt, num_inference_steps=50).images[0]
```

Follow [this guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/kerascv) to know more. 

## Notes on preparing data for DreamBooth training of faces

In addition to the tips and tricks shared in [this blog post](https://huggingface.co/blog/dreambooth#using-prior-preservation-when-training-faces), we followed these things while preparing the instances for conducting DreamBooth training on human faces:

* Instead of 3 - 5 images, use 20 - 25 images of the same person varying different angles, backgrounds, and poses.
* No use of images containing multiple persons. 
* If the person wears glasses, don't include images only with glasses. Combine images with and without glasses.

Thanks to [Abhishek Thakur](https://no.linkedin.com/in/abhi1thakur) for sharing these tips. 

## Acknowledgements

* Thanks to Hugging Face for providing the [original example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). It's very readable and easy to understand.
* Thanks to the ML Developer Programs' team at Google for providing GCP credits.
