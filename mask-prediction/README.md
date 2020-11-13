# Deep Sketch-Based Modeling: Tips and Tricks 

This repository contains the code for binary mask prediction, this code is based on [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf).

You can find detailed usage instructions for training and evaluation below.

 If you use our code or dataset, please cite our work:

    @inproceedings{deepsketch2020,
        title = {Deep Sketch-Based Modeling: Tips and Tricks },
        author = {Yue, Zhong and Yulia, Gryaditskaya and Honggang, Zhang and Yi-Zhe, Song},
        booktitle = {Proceedings of International Conference on 3D Vision (3DV)},
        year = {2020}
    }

## Installation
First you have to make sure that you have all dependencies in place.

Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

## Usage
When you have installed all binary dependencies and obtained the preprocessed data, you are ready to run our code.

### Training

To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

Train a model:
```
python train.py --name sketch2mask --model pix2pix --direction BtoA
```

You are able to change the dataset root by `--dataroot` command

### Generation
Test the model:
```
python test.py --name sketch2mask --model pix2pix --direction BtoA
```
The test results will be saved to a html file here: `./results/sketch2mask`. You can find more scripts at `scripts` directory.


### Evaluation
For evaluation of the models, we provide the evaluation script: `eval.py` .



You can run it using
```
python eval.py
```
The script takes the masks generated in the previous step and evaluates them using a standardized protocol. Change the `generation_path` inside to your resutls path for evaluation.
