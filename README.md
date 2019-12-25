# Decoupled Sub-Pixel Detection

## 1. Project description

Sub-pixel detection project in SJTU CS386. Our object is to detect interest points in sub-pixel accuracy. We proposed Decoupled Sub-Pixel Detection(DSPD), a novel formulation of Sub-Pixel Detection task.

The implementation is based on Python 3 and Pytorch 1.3.1. Experiments runs on Ubuntu and MacOS.

## 2. Usage

This part will run a simple demo of .

### 2.1. Install requirements

```bash
pip install -r requirements.txt
```

### 2.2. Prepare datasets

Put train HR(High-Resolution) images in `./data/intput/test_input_small`    We already put some demo images from DIV2K in the folder. If you want to use more images, DIV2K full dataset is available at  https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format . 

### 2.3. Download pre-trained model

Please download SAN model from https://github.com/daitao/SAN, and EDSR, MDSR model from https://github.com/thstkdgus35/EDSR-PyTorch. And put these model files in `./weight`.

### 2.4. Generate dataset

This step will split images into same shape, perform data augment and generate ground truth. 

```
python utils/dataPrepare.py
```

### 2.5. Evaluate models

This step will run DSPD model, and print confusion matrix. Defaultly, it uses EDSR model, with scale factor = 2. 

```
python eval.py
```

### 2.6. Check outputs

The SR images and detection images can be find in `./data/output`.

## 3.Remarks

Current code is based on our previous End-to-End approach in  https://github.com/FengMingquan-sjtu/spd.