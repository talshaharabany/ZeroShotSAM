# ZeroShotSAM

## Overview
ZeroShotSAM is a project focused on zero-shot medical image segmentation using a sparse prompt approach, leveraging a finetuned version of the Segment Anything Model (SAM). This repository contains code and resources for finetune and evaluating the model, as well as documentation to help users get started.

## Features
- Utilizes the Segment Anything Model (SAM) for medical image segmentation.
- Implements zero-shot segmentation using sparse prompts.
- Provides a finetuning mechanism to adapt SAM to medical domain data.
- Includes a script for finetuning and evaluation.

## Datasets

We used the following datasets in our experiments:

[monu](https://drive.google.com/drive/folders/1bzyHsDWhjhiwzpx_zJ5dpMG3-5F-nhT4?usp=drive_link)
[glas](https://drive.google.com/drive/folders/1z9xBesNhvuM08yUOpOWcUy7OnBGHenFv?usp=drive_link)

## SAM checkopints

[sam base](https://drive.google.com/file/d/1ZwKc-7Q8ZaHfbGVKvvkz_LPBemxHyVpf/view?usp=drive_link)
[sam large](https://drive.google.com/file/d/16AhGjaVXrlheeXte8rvS2g2ZstWye3Xx/view?usp=drive_link)

## Usage

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --task glas --vit vit_b --epoches 100
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py --task glas --vit vit_l --epoches 100
CUDA_VISIBLE_DEVICES=2 python -W ignore train.py --task monu --vit vit_b --epoches 100