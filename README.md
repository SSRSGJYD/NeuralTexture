# Neural Texture

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
  + tqdm



## File Organization

The root directory contains several subdirectories and files:

```
dataset/ --- custom PyTorch Dataset classes for loading included data
model/ --- custom PyTorch Module classes
util.py --- useful procedures
render.py --- render using texture and U-Net
render_texture.py --- render from RGB texture or neural texture
train.py --- optimize texture and U-Net jointly
train_texture.py --- optimize only texture
train_unet.py --- optimize U-Net using pretrained 3-channel texture
```



## How to Use

### Set up Environment

Install python >= 3.6 and create an environment.

Install requirements:

```powershell
pip install -r requirements.txt
```

### Prepare Data

We need 3 folders of data:

+ `/data/frame/`  with video frames `.png` files
+ `/data/uv/`  with uv-map `.npy` files, each shaped (H, W, 2)
+ `/data/extrinsics/`  with normalized camera extrinsics in  `.npy` files, each shaped (3)

Each frame corresponds to one uv map and corresponding camera extrinsic parameters. They are named sequentially, from `0000` to `xxxx` .

We demonstrate 2 ways to prepare data. One way is to render training data, the code is at https://github.com/A-Dying-Pig/OpenGL_NeuralTexture. The other way is to reconstruct from real scene, the code is at https://github.com/gerwang/InfiniTAM .

### Configuration

Rename `config_example.py` as `config.py` and set the parameters for training and rendering.

### Train Jointly

```powershell
python train.py [--args]
```

### Train Texture

```powershell
python train_texture.py [--args]
```

### Train U-Net

```powershell
python train_unet.py [--args]
```

### Render by Texture

```powershell
python render_texture.py [--args]
```

### Render by Texture and U-Net Jointly

```powershell
python render.py [--args]
```

