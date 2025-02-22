# GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images

This code is a **pytorch** implementation of our paper **"GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images"**.


## Visualization

<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/GL-LCM_gif.gif" width="50%">
</div>

## Primary Contributions

1) We introduce an **end-to-end LDM-based framework for high-resolution bone suppression**, named **BS-LDM**. It utilizes a **multi-level hybrid loss-constrained VQGAN for effective perceptual compression**. This framework consistently generates soft tissue images with high levels of bone suppression while preserving fine details and critical lung lesions.
    

2) To enhance the quality of generated images, we incorporate **offset noise** and a **temporal adaptive thresholding strategy**. These innovations help minimize discrepancies in low-frequency information, thereby improving the interpretability of the soft tissue images.

    
3) We have compiled a comprehensive bone suppression dataset, **SZCH-X-Rays**, which includes 818 pairs of high-resolution CXR and DES soft tissue images from our partner hospital. Additionally, we processed 241 pairs of images from the **JSRT** dataset into negative formats more commonly used in clinical settings.

4) Our **clinical evaluation** focused on image quality and diagnostic utility. The results demonstrated excellent image quality scores and substantial diagnostic improvements, underscoring the clinical significance of our work.
    

## Proposed Method

<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/frame.png" width="80%">
</div>

Overview of the proposed BS-LDM: (a) The training process of BS-LDM, where CXR and noised soft tissue data in the latent space are transmitted to the noise estimator for offset noise prediction and L2 loss calculation; (b) The training process of ML-VQGAN, where a multi-level hybrid loss-constrained VQGAN is used to construct a latent space by training the reconstruction of CXR and soft tissue images, using a codebook to represent the discrete features of the images; (c) The sampling process of BS-LDM, where the latent variables obtained after each denoising step are clipped using a temporal adaptive thresholding strategy for the sake of contrast stability.

## Visualization of high-frequency and low-frequency feature decomposition of latent variables before and after Gaussian noise addition using Discrete Fourier Transform
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/freq.png" width="80%">
</div>

## Power spectral densities of soft tissue images in SZCH-X-Rays, corresponding latent variables and Gaussian noise on 201 spectrogram components, averaged over 10000 samples
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/PSD.svg" width="80%">
</div>

## Illustration of the composition of offset noise
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/off.png" width="80%">
</div>

## Visualization of soft tissue images on SZCH-X-Rays and JSRT datasets produced by different methods
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/comp.png" width="80%">
</div>

## Visualization of ablation studies of offset noise and the temporal adaptive thresholding strategy on BS-LDM, with histograms given to visualize the pixel intensity distribution more intuitively
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/abl.png" width="80%">
</div>


## Pre-requisties
* Linux

* Python>=3.7

* NVIDIA GPU (memory>=6G) + CUDA cuDNN

### Pre-trained models
[VQGAN - SZCH-X-Rays](https://drive.google.com/file/d/1KcVK0F7lG5L9Zc0-pWPS9pucAWIG3yFc/view?usp=sharing)
[UNet - SZCH-X-Rays](https://drive.google.com/file/d/1zt5rV-d5wXVXCOgYqqM3C3r4wap6XkBe/view?usp=sharing)
[VQGAN - JSRT](https://drive.google.com/file/d/17qp7H3v6L4fOqZJCTWifpzXGydEQSloU/view?usp=sharing)
[UNet - JSRT](https://drive.google.com/file/d/12b2rykq6lw1hajEbMJtidJZRVl-ZXX3a/view?usp=sharing)


### Download the dataset
The original JSRT dataset and precessed JSRT dataset are located at [https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing](https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing) and [https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing](https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing), respectively.

Three paired images with CXRs and DES soft-tissues images of SZCH-X-Rays for testing are located at
```
└─BS-LDM
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
To do the evaluation process of VQGAN, please run the following command:
```
python vq-gan_eval.py
```      
To do the evaluation process of the conditional latent diffusion model, please run the following command:
```
python ldm_eval.py
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
Then after finishing the training of VQGAN, you can use the saved VQGAN model as a decoder when training the conditional latent diffusion model by running the following command:
```
python ldm_train.py
```

## Metrics
You can also run the following command about evaluation metrics in our experiment including BSR, MSE, PSNR and LPIPS:
```
python metrics.py
```

## Citation
```
@article{sun2024bs,
  title={BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models},
  author={Sun, Yifei and Chen, Zhanghao and Zheng, Hao and Ge, Ruiquan and Liu, Jin and Min, Wenwen and Elazab, Ahmed and Wan, Xiang and Wang, Changmiao},
  journal={arXiv preprint arXiv:2412.15670},
  year={2024}
}
```
