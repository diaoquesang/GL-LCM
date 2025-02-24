# GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images

This code is a **pytorch** implementation of our paper **"GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images"**.


## Visualization before and after bone suppression

<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/GL-LCM_gif.gif" width="50%">
</div>

## Primary Contributions
To overcome these challenges, we propose Global-Local Latent Consistency Model (GL-LCM). This is a novel framework for fast high-resolution bone suppression in CXR images based on Latent Consistency Models (LCMs). Our key contributions are summarized as follows:

1) The GL-LCM architecture facilitates **effective bone suppression** while retaining **texture details**. This is achieved through the design of **dual-path sampling** in the latent space combined with **global-local fusion** in the pixel space. 

2) GL-LCM significantly enhances inference efficiency, which requires only approximately **10%** of the inference time of current diffusion-based methods, making it more suitable for clinical applications.
    
3) We introduce **Local-Enhanced Guidance (LEG)** to mitigate potential **boundary artifacts** and **detail blurring** issues in local-path sampling, **without additional training**.

4) Extensive experiments on both the self-collected dataset **SZCH-X-Rays** and the public dataset **JSRT** demonstrate exceptional performance and efficiency of our GL-LCM.
    

## Proposed Method

<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/framework.png" width="100%">
</div>

Overview of GL-LCM framework. (a) Lung segmentation in the pixel space, (b) Dual-path sampling in the latent space, and (c) Global-local fusion in the pixel space.

## Comparisons
<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/comparison.png" width="100%">
</div>

## Ablation Study
<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/ablation.png" width="100%">
</div>

## Pre-requisties
* Linux

* Python>=3.7

* NVIDIA GPU (memory>=32G) + CUDA cuDNN

### Pre-trained models
[VQGAN - SZCH-X-Rays](https://drive.google.com/file/d/1L6M_2AoOgtPhi1B-klUKFUruyk__-_E-/view?usp=drive_link)
[UNet - SZCH-X-Rays](https://drive.google.com/file/d/1mrFJZ3M7aECe3weZvYF5G-DJYPc16zAi/view?usp=drive_link)
[VQGAN - JSRT](https://drive.google.com/file/d/1sJv4iyUGo2NHMUOscWBtON1Rd7jm22QF/view?usp=drive_link)
[UNet - JSRT](https://drive.google.com/file/d/1d1DmdHy84vWHGJs-74pMKRKfY66_4Dmx/view?usp=drive_link)


### Download the dataset
The original JSRT dataset and precessed JSRT dataset are located at [https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing](https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing) and [https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing](https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing), respectively.

Three paired images with CXRs and DES soft-tissues images of SZCH-X-Rays for testing are located at
```
└─data
    ├─ CXR
    │   ├─ 0.png
    │   ├─ 1.png
    │   └─ 2.png
    └─ BS
        ├─ 0.png
        ├─ 1.png
        └─ 2.png
```

### Install dependencies
```
pip install -r requirements.txt
```

## Evaluation
To do the evaluation process of VQGAN for visualization, please run the following command:
```
python vq-gan_eval.py
```      
To do the evaluation process of GL-LCM, please run the following command:
```
python batch_lcm_eval.py
```

## Training
If you want to train our model by yourself, you are primarily expected to split the whole dataset into training, validation and testing sets. Please run the following command:
```
python dataSegmentation.py
```
Then, you can run the following command to train the VQGAN model:
```
python vq-gan_train.py
```
Then after finishing the training of VQGAN, you can use the saved VQGAN model when training the noise estimator network of GL-LCM by running the following command:
```
python lcm_train.py
```

## Metrics
You can also run the following command about evaluation metrics including BSR, MSE, PSNR and LPIPS:
```
python metrics.py
```
