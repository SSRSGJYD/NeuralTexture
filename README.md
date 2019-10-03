# NeuralTexture



## Introduction

This repository implements [Deferred Neural Rendering: Image Synthesis using Neural Textures](https://arxiv.org/abs/1904.12356) .



## Requirements

+ Python 3.6+
  + argparse
  + NumPy
  + Pillow
  + pytorch
  + tensorboardX
  + torchvision



## File Organization

The root directory contains several subdirectories and files:

```
dataset/ --- custom PyTorch Dataset classes for loading included data
model/ --- custom PyTorch Module classes
program/ --- small auxiliary programs
util.py --- useful procedures
train.py --- main training script
```



## How to Use

To train the model, put uv-map `.npy` files and video frames `.ppm` files into one folder, set `DATA_DIR` in `config.py` , and run the command:

```powershell
python train.py [--args]
```

