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
- [4/1] We release the code and datasets of HADES.

## Prepare

HADES is based on LLaVA 1.5, PixArt, and the evaluation of HADES is based on Beaver-7b. You can download the corresponding weights from the following Huggingface space by cloning the repository using git-lfs.

|                              HADES Base: LLaVA 1.5 Weights                             |                            HADES Base: PixArt XL 2-1024-MS Weights                            |                          Evaluation Base: Beaver-7b Weights                          |
|:--------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|
| [Download](https://huggingface.co/liuhaotian/llava-v1.5-7b) | [Download](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | [Download](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0) |


Then you can copy the weights folder to `./checkpoint`


## Build HADES

1. Generate Dataset in HADES

```Shell
bash ./generate_benchmark.sh
```

2. Amplifying Image Harmfulness with LLMs
   
```Shell
bash ./amplifying_toxic.sh
```

3. Amplifying Image Harmfulness with Gradient Update
   
```Shell
bash ./white_box.sh
```

## Evaluation

Now you can use built HADES to evaluate MLLMs and evaluate their safaty alignment by running:

```Shell
bash run_evaluation.sh abstract gpt4v hades
```

The script will report the Attack Success Rate (ASR) associated with GPT-4V with our HADES. 


Furthermore, the `run_evaluation.sh` script can be also employed to calculate the ASR of HADES on other models such as LLaVA and Gemini.

## Related Projects

- [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA)
- [Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)
- [BeaverTails](https://github.com/PKU-Alignment/beavertails)

