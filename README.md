<div align="center">

![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:2193b0,100:45ed3F&text=GL-LCM:%20Global-Local%20Latent%20Consistency%20Models%20for%20Fast-nl-High-Resolution%20Bone%20Suppression%20in%20Chest%20X-Ray%20Images&reversal=false&fontSize=25&fontAlignY=30&desc=Yifei%20Sun,%20Zhanghao%20Chen,%20Hao%20Zheng,%20Yuqing%20Lu,%20Lixin%20Duan,%20Fenglei%20Fan,%20Ahmed%20Elazab,%20Xiang%20Wan,%20Changmiao%20Wang,%20Ruiquan%20Ge&descSize=12&descAlignY=55&fontColor=FFFFFF)

</div>

This code is a **pytorch** implementation of our paper "**GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images**" accepted by [MICCAI 2025](https://conferences.miccai.org/2025/en/default.asp). [[🌐 Project Page]](https://diaoquesang.github.io/GL-LCM-Project-Page/)


## 🎥 Visualization before (left) and after (right) bone suppression using GL-LCM

<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/GL-LCM_gif.gif" width="50%">
</div>

## 💡 Primary contributions
To overcome these challenges, we propose **Global-Local Latent Consistency Model (GL-LCM)**. This is a novel framework for **fast high-resolution bone suppression in CXR images** based on **Latent Consistency Models (LCMs)**. Our key contributions are summarized as follows:

1) 🕐The GL-LCM architecture facilitates **effective bone suppression** while retaining **texture details**. This is achieved through the design of **dual-path sampling** in the latent space combined with **global-local fusion** in the pixel space. 

2) 🕑GL-LCM significantly enhances inference efficiency, which requires only approximately **10%** of the inference time of current diffusion-based methods, making it more suitable for clinical applications.
    
3) 🕒We introduce **Local-Enhanced Guidance (LEG)** to mitigate potential **boundary artifacts** and **detail blurring** issues in local-path sampling, **without additional training**.

4) 🕓Extensive experiments on both the self-collected dataset **SZCH-X-Rays** and the public dataset **JSRT** demonstrate exceptional performance and efficiency of our GL-LCM.
    

## 🧗 Proposed method

<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/framework.png" width="100%">
</div>

Overview of GL-LCM framework. (a) Lung segmentation in the pixel space, (b) Dual-path sampling in the latent space, and (c) Global-local fusion in the pixel space.

## 🏎️ Comparisons

### - Qualitative Results on SZCH-X-Rays and JSRT
<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/comparison.png" width="100%">
</div>

### - Quantitative results on SZCH-X-Rays
<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>BSR (%)↑</th>
      <th>MSE (10⁻³)↓</th>
      <th>PSNR↑</th>
      <th>LPIPS↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5">Universal Method</td>
    </tr>
    <tr>
      <td>VAE</td>
      <td>91.281 ± 3.088</td>
      <td>1.169 ± 1.059</td>
      <td>30.018 ± 2.007</td>
      <td>0.237 ± 0.047</td>
    </tr>
    <tr>
      <td>VQ-VAE</td>
      <td>94.485 ± 2.407</td>
      <td>0.645 ± 0.596</td>
      <td>32.600 ± 2.071</td>
      <td>0.137 ± 0.029</td>
    </tr>
    <tr>
      <td>VQGAN</td>
      <td>94.330 ± 3.402</td>
      <td>0.923 ± 2.478</td>
      <td>32.096 ± 2.420</td>
      <td>0.083 ± 0.020</td>
    </tr>
    <tr>
      <td colspan="5">Task-Specific Method</td>
    </tr>
    <tr>
      <td>Gusarev et al.</td>
      <td>94.142 ± 2.666</td>
      <td>1.028 ± 2.201</td>
      <td>31.369 ± 2.385</td>
      <td>0.156 ± 0.031</td>
    </tr>
    <tr>
      <td>MCA-Net</td>
      <td>95.442 ± 2.095</td>
      <td>0.611 ± 0.435</td>
      <td>32.689 ± 1.939</td>
      <td>0.079 ± 0.018</td>
    </tr>
    <tr>
      <td>ResNet-BS</td>
      <td>94.508 ± 1.733</td>
      <td>0.646 ± 0.339</td>
      <td>32.265 ± 1.635</td>
      <td>0.107 ± 0.022</td>
    </tr>
    <tr>
      <td>Wang et al.</td>
      <td>89.767 ± 6.079</td>
      <td>1.080 ± 0.610</td>
      <td>29.963 ± 1.378</td>
      <td>0.072 ± 0.016</td>
    </tr>
    <tr>
      <td>BS-Diff</td>
      <td>92.428 ± 3.258</td>
      <td>0.947 ± 0.510</td>
      <td>30.627 ± 1.690</td>
      <td>0.212 ± 0.041</td>
    </tr>
    <tr>
      <td>BS-LDM</td>
      <td>94.159 ± 2.751</td>
      <td>0.701 ± 0.293</td>
      <td>31.953 ± 1.969</td>
      <td>0.070 ± 0.018</td>
    </tr>
    <tr>
      <td><strong>GL-LCM (Ours)</strong></td>
      <td><strong>95.611 ± 1.529</strong></td>
      <td><strong>0.512 ± 0.293</strong></td>
      <td><strong>33.347 ± 1.829</strong></td>
      <td><strong>0.056 ± 0.015</strong></td>
    </tr>
  </tbody>
</table>

### - Quantitative results on JSRT
<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>BSR (%)↑</th>
      <th>MSE (10⁻³)↓</th>
      <th>PSNR↑</th>
      <th>LPIPS↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5">Universal Method</td>
    </tr>
    <tr>
      <td>VAE</td>
      <td>85.646 ± 9.327</td>
      <td>1.224 ± 0.749</td>
      <td>29.814 ± 2.364</td>
      <td>0.155 ± 0.032</td>
    </tr>
    <tr>
      <td>VQ-VAE</td>
      <td>86.445 ± 8.881</td>
      <td>0.986 ± 0.596</td>
      <td>30.712 ± 2.273</td>
      <td>0.062 ± 0.017</td>
    </tr>
    <tr>
      <td>VQGAN</td>
      <td>86.594 ± 8.916</td>
      <td>1.002 ± 0.606</td>
      <td>30.635 ± 2.255</td>
      <td>0.061 ± 0.017</td>
    </tr>
    <tr>
      <td colspan="5">Task-Specific Method</td>
    </tr>
    <tr>
      <td>Gusarev et al.</td>
      <td>89.283 ± 8.288</td>
      <td>0.821 ± 0.570</td>
      <td>31.700 ± 2.594</td>
      <td>0.100 ± 0.024</td>
    </tr>
    <tr>
      <td>MCA-Net</td>
      <td>86.887 ± 9.825</td>
      <td>0.876 ± 0.625</td>
      <td>31.577 ± 2.905</td>
      <td>0.057 ± 0.017</td>
    </tr>
    <tr>
      <td>ResNet-BS</td>
      <td>88.782 ± 8.905</td>
      <td>0.960 ± 0.661</td>
      <td>31.021 ± 2.576</td>
      <td>0.060 ± 0.016</td>
    </tr>
    <tr>
      <td>Wang et al.</td>
      <td>89.679 ± 9.477</td>
      <td>1.013 ± 0.655</td>
      <td>30.681 ± 2.431</td>
      <td>0.075 ± 0.015</td>
    </tr>
    <tr>
      <td>BS-Diff</td>
      <td>88.707 ± 8.859</td>
      <td>1.003 ± 0.655</td>
      <td>30.765 ± 2.504</td>
      <td>0.154 ± 0.037</td>
    </tr>
    <tr>
      <td>BS-LDM</td>
      <td>89.322 ± 9.562</td>
      <td>0.783 ± 0.632</td>
      <td>32.307 ± 3.231</td>
      <td>0.058 ± 0.017</td>
    </tr>
    <tr>
      <td><strong>GL-LCM (Ours)</strong></td>
      <td><strong>90.056 ± 10.635</strong></td>
      <td><strong>0.746 ± 0.680</strong></td>
      <td><strong>32.951 ± 3.799</strong></td>
      <td><strong>0.052 ± 0.015</strong></td>
    </tr>
  </tbody>
</table>

### - Inference efficiency comparison on SZCH-X-Rays
<table align="center">
  <tr>
    <th>Method</th>
    <th>Sampler</th>
    <th>Sampling Steps</th>
    <th>Parameters</th>
    <th>Inference Time (s)</th>
  </tr>
  <tr>
    <td>BS-Diff</td>
    <td>DDPM</td>
    <td>1000</td>
    <td>254.7M</td>
    <td>108.86</td>
  </tr>
  <tr>
    <td>BS-LDM</td>
    <td>DDPM</td>
    <td>1000</td>
    <td>421.3M</td>
    <td>84.62</td>
  </tr>
  <tr>
    <td><strong>GL-LCM (Ours)</strong></td>
    <td><strong>LCM</strong></td>
    <td><strong>50</strong></td>
    <td><strong>436.9M</strong></td>
    <td><strong>8.54</strong></td>
  </tr>
</table>

## 🙇 Ablation study
### - Qualitative results of LEG on SZCH-X-Rays and JSRT
<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/ablation.png" width="100%">
</div>

A pseudo-color zoomed-in view is shown in the bottom right corner, and the green arrows mark the boundary artifacts.

### - Quantitative results of LEG on SZCH-X-Rays and JSRT
<table align="center">
  <thead>
    <tr>
      <th rowspan="2">Guidance Method</th>
      <th colspan="2">SZCH-X-Rays</th>
      <th colspan="2">JSRT</th>
    </tr>
    <tr>
      <th>PSNR↑</th>
      <th>LPIPS↓</th>
      <th>PSNR↑</th>
      <th>LPIPS↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Vanilla Guidance</td>
      <td>32.777 ± 2.091</td>
      <td>0.058 ± 0.016</td>
      <td>32.296 ± 3.454</td>
      <td>0.073 ± 0.020</td>
    </tr>
    <tr>
      <td>CFG</td>
      <td>32.315 ± 1.717</td>
      <td>0.068 ± 0.013</td>
      <td>32.613 ± 3.604</td>
      <td>0.070 ± 0.015</td>
    </tr>
    <tr>
      <td><strong>LEG (Ours)</strong></td>
      <td><strong>33.347 ± 1.829</strong></td>
      <td><strong>0.056 ± 0.015</strong></td>
      <td><strong>32.951 ± 3.799</strong></td>
      <td><strong>0.052 ± 0.015</strong></td>
    </tr>
  </tbody>
</table>

### - Quantitative results of Poisson Fusion on SZCH-X-Rays and JSRT
<table align="center">
  <tr>
    <th rowspan="2">Fusion Strategy</th>
    <th colspan="2">SZCH-X-Rays</th>
    <th colspan="2">JSRT</th>
  </tr>
  <tr>
    <th>PSNR↑</th>
    <th>LPIPS↓</th>
    <th>PSNR↑</th>
    <th>LPIPS↓</th>
  </tr>
  <tr>
    <td>✗</td>
    <td>31.360 ± 2.079</td>
    <td>0.091 ± 0.020</td>
    <td>31.638 ± 3.078</td>
    <td>0.074 ± 0.021</td>
  </tr>
  <tr>
    <td>α-Fusion</td>
    <td>29.781 ± 1.522</td>
    <td>0.181 ± 0.021</td>
    <td>31.784 ± 3.043</td>
    <td>0.092 ± 0.013</td>
  </tr>
  <tr>
    <td>AE Fusion</td>
    <td>30.850 ± 1.806</td>
    <td>0.141 ± 0.028</td>
    <td>31.835 ± 3.075</td>
    <td>0.061 ± 0.017</td>
  </tr>
  <tr>
    <td><strong>Poisson Fusion (Ours)</strong></td>
    <td><strong>33.347 ± 1.829</strong></td>
    <td><strong>0.056 ± 0.015</strong></td>
    <td><strong>32.951 ± 3.799</strong></td>
    <td><strong>0.052 ± 0.015</strong></td>
  </tr>
</table>

## ⚙️ Pre-requisties
* Linux

* Python>=3.7

* NVIDIA GPU (memory>=32G) + CUDA cuDNN

## 🚀 Pre-trained models
[VQGAN - SZCH-X-Rays](https://drive.google.com/file/d/1L6M_2AoOgtPhi1B-klUKFUruyk__-_E-/view?usp=drive_link)
[UNet - SZCH-X-Rays](https://drive.google.com/file/d/1mrFJZ3M7aECe3weZvYF5G-DJYPc16zAi/view?usp=drive_link)
[VQGAN - JSRT](https://drive.google.com/file/d/1sJv4iyUGo2NHMUOscWBtON1Rd7jm22QF/view?usp=drive_link)
[UNet - JSRT](https://drive.google.com/file/d/1d1DmdHy84vWHGJs-74pMKRKfY66_4Dmx/view?usp=drive_link)


## 📦 Download the datasets
The original JSRT dataset and processed JSRT dataset are located at [https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing](https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing) and [https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing](https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing), respectively.

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
To implement lung segmentation in data preparation, please use [lungSegmentation.ipynb](https://github.com/diaoquesang/GL-LCM/blob/main/codes/lungSegmentation.ipynb).

## 🫳 Install dependencies
```
pip install -r requirements.txt
```

## 🚅 Evaluation
To do the evaluation process of VQGAN for visualization, please run the following command:
```
python vq-gan_eval.py
```      
To do the evaluation process of GL-LCM, please run the following command:
```
python batch_lcm_eval.py
```

## 🐎 Training
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

## 🔍 Metrics
You can also run the following command about evaluation metrics including BSR, MSE, PSNR and LPIPS:
```
python metrics.py
```
