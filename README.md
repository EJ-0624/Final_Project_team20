# Final_Project_team20

## Environment

* Ubuntu 18.04.5 LTS
* torch 1.10.0
* cuda 11.1

```setup
# Check cuda -version
nvcc -V
nvidia-smi
```

## Requirements

Install the corresponding cuda version of pytorch
```setup
# CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install mmcv
https://github.com/open-mmlab/mmcv

```setup
# method 1
# Note: mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well. For example, if your PyTorch version is 1.8.1 and CUDA version is 11.1, you can use the following command to install mmcv-full.
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# method 2
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# method 3
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install .
```

Install mmdetection
```setup
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
## Train image
![image](https://user-images.githubusercontent.com/68366624/148245902-353230a1-c1a0-42e7-b260-a650a4c2cf4b.png)

* Dataset: 26,684 train, 3000 test

![image](https://user-images.githubusercontent.com/68366624/148247990-bce17382-fddd-4cef-8b1c-1812463002b1.png)

## Relate Work

![image](https://user-images.githubusercontent.com/68366624/148248420-65815157-e961-46d6-9f36-a8dd4817bdce.png)

## Model 1 for Efficientnet-b2
```setup
python rsna_efficientnet_b2.py
```
## Model 1 result

![image](https://user-images.githubusercontent.com/68366624/148248981-7acaea7b-f13c-4c64-9a5d-a786c76248ea.png)
![image](https://user-images.githubusercontent.com/68366624/148248847-508da98f-7577-4e3e-9e31-b0d385f41128.png)

## Model 2 for objectdetection
```setup
python tools/train.py retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_rsna.py --work-dir retinanet_r101_fpn_2x_coco
```

## Model 2 log analysis

```setup
python ./tools/analysis_tools/analyze_logs.py plot_curve retinanet_r101_fpn_2x_coco/20220104_211554.log.json --keys bbox_mAP --legend loss_cls --out losses.pdf
```

## Model 2 result

```setup
python ./tools/test.py  retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_rsna.py retinanet_r101_fpn_2x_coco/epoch_30.pth --out result_retinanet/result.pkl --eval bbox
```


