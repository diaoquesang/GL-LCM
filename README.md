# GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images

This code is a **pytorch** implementation of our paper **"GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images"**.


## Visualization before and after bone suppression

<div align="center">
<img src="https://github.com/diaoquesang/GL-LCM/blob/main/images/GL-LCM_gif.gif" width="50%">
</div>

## Primary Contributions
To overcome these challenges, we propose **Global-Local Latent Consistency Model (GL-LCM)**. This is a novel framework for **fast high-resolution bone suppression in CXR images** based on **Latent Consistency Models (LCMs)**. Our key contributions are summarized as follows:

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
      <td>GL-LCM (Ours)</td>
      <td>95.611 ± 1.529</td>
      <td>0.512 ± 0.293</td>
      <td>33.347 ± 1.829</td>
      <td>0.056 ± 0.015</td>
    </tr>
  </tbody>
</table>
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
