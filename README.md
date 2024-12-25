# few-shot-annotation

This repo is the aggregation of [devit](https://github.com/mlzxy/devit/tree/main) repo for simplifying the yolo annotation.

## Installation
Clone and install DE-ViT repo and dependencies.
```bash
git clone https://github.com/mlzxy/devit.git
python3 -m venv venv_devit
source venv_devit/bin/activate
pip install -r devit/requirements.txt
pip install -e ./devit
pip install -r requirements.txt
```

## Downloads
### DEV-iT 
You need to download weights of models, please refer to [Downloads.md](https://github.com/mlzxy/devit/blob/main/Downloads.md).
The weights archive is pretty big(25.3 GB.), so you might want to download only specific weights. After the weights are downloaded, move the folder to the root of this project directory saving the original structure.

```bash
├── weights/
│   └── initial/
│   └── trained/
```

### SAM
If you don't have the mask annotation for your samples, you can generate them with [SAM](https://github.com/facebookresearch/segment-anything). For using the SAM you need to download checkpoint(s).
```bash
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth # for ViT-H
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth # for ViT-L
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth # for ViT-B
``` 

## Usage

### Structure of yolo root:

```bash
├── yolo_root/
│   └── img_1.jpg
│   └── img_1.txt
```

Generate masks with yolo
```bash
python3 yolo_to_mask.py -i ../data/yolo_root -o ../data/generated
```

Generate prototypes with masks and source images
```bash
python3 build_prototypes.py --path ../data/generated --output test.pth
```

Run demo
```bash
python3 demo.py -i ../data/test -o ../data/test/output --category test.pth --overlapping
```

Run automatic annotation
```bash
python3 automatic_video_yolo_annotation.py -i /home/user/Videos/src.mp4 --category test.pth --enable_drawing
```
