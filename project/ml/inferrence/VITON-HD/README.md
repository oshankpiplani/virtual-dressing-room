## Installation

Clone this repository:

```
git clone https://github.com/oshankpiplani/virtual-dressing-room/edit/main/project/ml/inferrence/VITON-HD
cd ./VITON-HD/
```

Install PyTorch and other dependencies:

```
conda create -y -n [ENV] python=3.8
conda activate [ENV]
conda install -y pytorch=[>=1.6.0] torchvision cudatoolkit=[>=9.2] -c pytorch
pip install opencv-python torchgeometry
```

## Pre-trained networks

Please download `*.pkl` and test images from the [VITON-HD Google Drive folder](https://drive.google.com/drive/folders/0B8kXrnobEVh9fnJHX3lCZzEtd20yUVAtTk5HdWk2OVV0RGl6YXc0NWhMOTlvb1FKX3Z1OUk?resourcekey=0-OIXHrDwCX8ChjypUbJo4fQ&usp=sharing) and unzip `*.zip` files. `test.py` assumes that the downloaded files are placed in `./checkpoints/` and `./datasets/` directories.

## Testing

To generate virtual try-on images, run:

```
CUDA_VISIBLE_DEVICES=[GPU_ID] python test.py --name [NAME]
```

The results are saved in the `./results/` directory. You can change the location by specifying the `--save_dir` argument. To synthesize virtual try-on images with different pairs of a person and a clothing item, edit `./datasets/test_pairs.txt` and run the same command.



