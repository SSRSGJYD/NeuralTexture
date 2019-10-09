# NeuralTexture



## Introduction

This repository implements [Deferred Neural Rendering: Image Synthesis using Neural Textures](https://arxiv.org/abs/1904.12356) .



## Requirements

+ Python 3.6+
  + argparse
  + nni
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
render.py --- main render script
train.py --- main training script
```



## How to Use

### Set up Environment

Install python >= 3.6 and create an environment.

Install requirements:

```powershell
pip install -r requirements.txt
```

### Train

To train the model, put uv-map `.npy` files and video frames `.ppm` files into one folder, set parameters in `config.py` , and run the command:

```powershell
python train.py [--args]
```

### Train with AutoML

To use `nni` , change settings in `config.yaml` and `search_space.json` and run:

```powershell
nnictl create --config config.yml [--port 8088] [--debug] 
```

For detailed usage, check https://github.com/microsoft/nni.

### Render

To render images using trained model, put uv-map `.npy` files into one folder, set parameters in `config.py` , and run the command:

```powershell
python render.py [--args]
```

