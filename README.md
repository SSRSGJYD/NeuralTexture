# NeuralTexture

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
program/ --- small auxiliary programs
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
+ `/data/view_direction/`  with view direction map  `.npy` files,, each shaped (H, W, 3)

Each frame corresponds to one uv map and one view direction map. They are named sequentially, from `0000` to `xxxx` .

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

### Train Jointly with AutoML

To use `nni` , change settings in `config.yaml` and `search_space.json` and run:

```powershell
nnictl create --config config.yml [--port 8088] [--debug] 
```

For detailed usage, check https://github.com/microsoft/nni.

### Render by Texture

```powershell
python render_texture.py [--args]
```

### Render by Texture and U-Net Jointly

```powershell
python render.py [--args]
```

