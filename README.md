# Deep Sketch-Based Modeling: Tips and Tricks 

## Contents

- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Download Dataset](#Download-Dataset)
- [Results](#Results)

## Introduction

This repository contains the Pytorch implementation of [Deep Sketch-Based Modeling: Tips and Tricks](https://arxiv.org/abs/2011.06133), including binary mask prediction and 3D shape reconstruction. 

You can find detailed usage instructions for training and evaluation below.

 If you use our code or dataset, please cite our work:

    @inproceedings{deepsketch2020,
        title = {Deep Sketch-Based Modeling: Tips and Tricks },
        author = {Yue, Zhong and Yulia, Gryaditskaya and Honggang, Zhang and Yi-Zhe, Song},
        booktitle = {Proceedings of International Conference on 3D Vision (3DV)},
        year = {2020}
    }

## Requirements

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

Please refer the README file in each sub-task for detailed instruction.

## Download Dataset

Download dataset is easy. Directly run

```bash
sh scripts/download_dataset.sh
```

## Results

We identify key differences between sketch and image inputs, driving out important insights and proposing the respective solutions, we show an improved performance of deep image modeling..

<img src="img/tease.gif" width="700">

