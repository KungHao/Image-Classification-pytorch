---
title: 'Image Classification on pytorch framework'
disqus: hackmd
---

Image-Classification-pytorch
===

## Table of Contents

[TOC]

## Beginners Guide

If you are a total beginner to this, start here!

1. 
2. Click "Sign in"
3. Choose a way to sign in
4. Start writing note!

Preparing dataset
---

1. Download [Cartoon Set](https://google.github.io/cartoonset/index.html)  or [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)(Align&Cropped Images)
2. Configure dataset path in `./datasets/{dataset name}`
3. Make your directory like this:
```bash
├── datasets
│   ├── cartoon
│   │   ├── cartoon_folder
│   │   │   ├── 屬性名稱資料夾...
│   │   ├── images
│   │   ├── list_attr_cartoon.txt
│   ├── celeba
│   │   ├── images
│   │   ├── list_attr_celeba.txt
├── experiments
│   ├── checkpoints
│   ├── acc_figures
└── *.py
```

Cartoon Set Preprocess
---

**透過cartoonset_preprocess.py製作出如下圖的.txt屬性表**

![txt屬性表](https://i.imgur.com/5zNBAwt.png)

#### 注意事項:

1. 由csv轉換為txt要注意編碼為UTF-8。
2. 由於csv資料表間隔較大，因此要將所有間隔取代為正常大小空格。


Usage
---

執行前確認:
* num_epochs: number of epochs
* 是否預訓練: ImageNet 預訓練模式以及訓練卷積層
* figure名稱: 圖表的路徑和圖表名稱
* model名稱: 模型架構名稱 e.g. AlexNet, VGGNet13 參考utils.py內的initialize_model
* 資料集比例: 是否要用random_sampler，dataset.py內修改

執行train.py訓練classification model
執行test.py產出Confusion Matrix

Q&A
---

* Q1: Normalize正規化mean std數值。
* A1: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)，若使用0.5會使MAGAN訓練產出模糊圖像。
* Q2: 資料集正負比例差距過大。
* A2: 訓練失敗，網絡會太過專注於例子較多的case。
* Q3: 如何得到Crop_size。
* A3: 經過測試輸出圖像，使crop center能夠刪除最多背景圖。
* Q4: Image_siez大小如何決定。
* A4: 根據[Pytorch finetuning](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)實驗結果決定使用224。

###### tags: `Attribute discriminator` `Image classification`
