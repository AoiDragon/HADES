## Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models

This repo provides the source code & data of our paper: [Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models](https://arxiv.org/abs/2403.09792).

```bibtex
@article{Li-HADES-2024,
  author       = {Yifan Li and Hangyu Guo and Kun Zhou and Wayne Xin Zhao and Ji{-}Rong Wen},
  title        = {Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2403.09792},
  year         = {2024}
}
```

## Overview

In this paper, we study the harmlessness alignment problem of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named **HADES**, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision.

![model_figure](./figs/hades.jpg)

## Update
- [9/26] 🤗 We release HADES benchmark on Hugging Face. Feel free to evaluate your MLLMs by visiting [https://huggingface.co/datasets/Monosail/HADES](https://huggingface.co/datasets/Monosail/HADES)
- [8/12] 🎉🎉🎉 HADES is selected as an oral paper of ECCV 2024!
- [7/4] 🎉🎉🎉 HADES is accepted by ECCV 2024!
- [5/5] We release the original HADES-generated-data in our paper.
- [4/10] We release the code of HADES.

## Dataset

We are excited to release the data we used in our paper generated by HADES, which can be accessed at [this link](https://drive.google.com/drive/folders/1k4coKdLd_iLhwyTyWmz8nQ0thN4D01qc?usp=sharing). You can use these data to reproduce our experiment results and you can also use HADES to build your own data by following the guidance in the [following part](#preparation).
The dataset is structured as follows:
- `/instructions`: The text instructions in 5 harmful scenarios.
- `/google_images`: The images retrieved from Google and filtered by CLIP.
- `/typo_images`: The typography images of the harmful keywords in instructions.
- `/optimized_SD_images`:  Images generated through diffusion models and further optimized using LLMs. Each image is labeled with 'opt_{step}.jpg', where 'step' denotes the optimization step; step=0 corresponds to the initial image. Additionally, we provide concatenated diffusion images along with their corresponding typography images.
- `/llava_15_7b_images`: Concatenations of adversarial images updated on LLaVA-1.5 7b, optimized diffusion images, and their associated typography images.
  
### Safety Declaration:
While we release this dataset for research purposes, we emphasize the importance of using it responsibly and ethically. The data contained within the HADES dataset may depict or pertain to sensitive or harmful subjects. **By accessing the HADES dataset, you agree not to use the data for any illegal or harmful activities. Please ensure responsible and ethical use at all times.**


## Preparation
HADES is based on LLaVA 1.5, PixArt, and the evaluation of HADES is based on Beaver-7b. You can download the corresponding weights from the following Huggingface space by cloning the repository using git-lfs.

|                              HADES Base: LLaVA 1.5 Weights                             |                            HADES Base: PixArt XL 2-1024-MS Weights                            |                          Evaluation Base: Beaver-7b Weights                          |
|:--------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|
| [Download](https://huggingface.co/liuhaotian/llava-v1.5-7b) | [Download](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | [Download](https://huggingface.co/PKU-Alignment/beaver-dam-7b) |


Then you can copy the weights folder to `./checkpoint`


## Harmful Data Collection using HADES 

1. Generating harmful text instructions

```Shell
bash ./generate_benchmark.sh
```

2. Amplifying image harmfulness with LLMs
   
```Shell
bash ./amplifying_toxic.sh
```

3. Amplifying image harmfulness with gradient update
   
```Shell
bash ./white_box.sh
```

## Evaluation

Now you can use the collected images and text to evaluate the safaty alignment of MLLMs by running the following script. The 'abstract' parameter refers to the 'text-to-image pointer' setting in our paper.

```Shell
bash run_evaluation.sh abstract gpt4v hades
```

The script will report the Attack Success Rate (ASR) associated with GPT-4V with our HADES in abstract settings. 

```Shell
bash run_evaluation.sh abstract gpt4v black_box
```

And you can run the script to obtain the ASR on GPT-4V with our HADES without a white-box attack.

Furthermore, the `run_evaluation.sh` script can be also employed to calculate the ASR of HADES on other models such as LLaVA and Gemini. 

## Related Projects

- [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA)
- [Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)
- [BeaverTails](https://github.com/PKU-Alignment/beavertails)

