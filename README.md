## Temporal Convolutional Feature Pyramid Network (TCFPN) &amp; <br> Iterative Soft Boundary Assignment (ISBA)

### Introduction

In this work, we address the task of weakly-supervised human action segmentation in long, untrimmed videos. We propose a novel action modeling framework, which consists of a new temporal convolutional network, named Temporal Convolutional Feature Pyramid Network (TCFPN), for predicting frame-wise action labels, and a novel training strategy for weakly-supervised sequence modeling, named Iterative Soft Boundary Assignment (ISBA), to align action sequences and update the network in an iterative fashion. Details can be found in **[Weakly-Supervised Action Segmentation with Iterative Soft Boundary Assignment](https://arxiv.org/abs/1803.10699) (Li Ding & Chenliang Xu, CVPR '18)**

This repo includes Keras + Tensorflow implementation on [Breakfast](http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) dataset. Tested with Ubuntu 16.04 + Python 2.7.

---

### Overview

- `train_spv.py`  - supervised training and testing using ED-TCN / TCFPN

- `train_weak.py` - weakly-supervised training and testing using ED-TCN / TCFPN + ISBA

- `utils/...`     - model definitions, metrics, utils

- `data/...`      - intruction to get the data

---

### Quick Start

Please follow the instruction in [`data/README`](./data/README) to obtain the data, then run

`python train_spv.py` for supervised experiments on Breakfast dataset 

or

`python train_weak.py` for weakly-supervised experiments on Breakfast dataset.

---

### Citing
If you find TCFPN / ISBA useful in your research, please consider citing:

```
@InProceedings{Ding_2018_CVPR,
author = {Ding, Li and Xu, Chenliang},
title = {Weakly-Supervised Action Segmentation with Iterative Soft Boundary Assignment},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2018}
}
```
