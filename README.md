# Shadow-Aware Dynamic Convolution for Shadow Removal

## Introduction

We propose a Shadow-Aware Dynamic Convolution (SADC) to resolve the contradiction between the shadow region and non-shadow region for shadow removal. Please refer to the paper for details.

<img src=".\images\sadc.png" alt="image-20220405145732634" style="zoom:100%;" />

## Dataset

- ISTD [[Download](https://github.com/DeepInsight-PCALab/ST-CGAN)], ISTD Generated Mask [[link](https://github.com/rayleizhu/FDRNet)]
- SRD [Please email [author](http://vision.sia.cn/our%20team/JiandongTian/JiandongTian.html) to get download link]

## Pretrained Models & Testing Results

We conduct experiments on the ISTD dataset and SRD dataset without reshaping, ***i.e.***, 640x480 for the ISTD dataset and 840x640 for the SRD dataset.

- ISTD [[Download](https://drive.google.com/drive/folders/1O7WdOARj3A5CFhWu2Yvz8OpAbBXyDJc4?usp=sharing)], ISTD Results [[Download](https://drive.google.com/drive/folders/1TNd-y_th2JuW_UK-pyJK0cPRe5cKzIDX?usp=sharing)]
- SRD [[Download](https://drive.google.com/drive/folders/1-vZoh27t3Bl1-TZfhW600BufFYW4_-d3?usp=sharing)], SRD Results [[Download](https://drive.google.com/drive/folders/1yLeavgUzw0_EjqgttkLKv8AzfK-zduCn?usp=sharing)]

## Training

Modify the **dataset path** and **experiments name** in file `script/train.sh` and run the following script

```shell
cd script
sh ./train.sh GPU_ID VISDOM_PORT_ID
```

The usage of **visdom** can be referred to [link](https://github.com/fossasia/visdom). Once visdom is enabled in the python environment, the visualization of  the training can be referred to `http://server_id/visdom_port_id`.

## Testing

Similar to the training phase, please modify the **dataset path** and **experiments name** in file `script/test.sh` and run the following script, and make sure the name used in testing exists in the checkpoint folder. Testing MATLAB code is borrowed from [G2R Network](https://github.com/hhqweasd/G2R-ShadowNet).

```shell
cd script
sh ./test.sh GPU_ID
```

##### Note: the testing size for the ISTD dataset and SRD dataset is different, please modify the size to match the input size. The SRD dataset requires large RAM of GPU when testing, please switch to CPU testing (as we do in our experiments) if no such large GPU is available.

## Visualization

The comparison between our method and current SOTA methods is shown as below,

<img src=".\images\results.png" alt="image-20220405153110367" style="zoom: 200%;" />

## Ref

Link to the pre-printed paper: https://arxiv.org/abs/2205.04908

Author email: xuyimin0926@gmail.com
