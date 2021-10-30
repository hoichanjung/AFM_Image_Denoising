# Atomic Force MicroScope(AFM) Image Denoising

## Requirements
- Python 3.7.0
- Cuda 11.1
- pip install -r requirements.txt

## Data Description
1. 652 Dummy AFM Images with 3 Types of Noise (Random/Line/Scar)
2. AFM Image collected from SKHynix (by Seung Jun JUNG)

## Model
**- HINet: Half Instance Normalization Network for Image Restoration****
**- Paper: https://arxiv.org/abs/2105.06086**
In this paper, we explore the role of Instance Normalization in low-level vision tasks. Specifically, we present a novel block: Half Instance Normalization Block (HIN Block), to boost the performance of image restoration networks. Based on HIN Block, we design a simple and powerful multi-stage network named HINet, which consists of two subnetworks. With the help of HIN Block, HINet surpasses the state-of-the-art (SOTA) on various image restoration tasks. For image denoising, we exceed it 0.11dB and 0.28 dB in PSNR on SIDD dataset, with only 7.5% and 30% of its multiplier-accumulator operations (MACs), 6.8 times and 2.9 times speedup respectively. For image deblurring, we get comparable performance with 22.5% of its MACs and 3.3 times speedup on REDS and GoPro datasets. For image deraining, we exceed it by 0.3 dB in PSNR on the average result of multiple datasets with 1.4 times speedup. With HINet, we won 1st place on the NTIRE 2021 Image Deblurring Challenge - Track2. JPEG Artifacts, with a PSNR of 29.70.

### Network Architecture
![image](https://user-images.githubusercontent.com/59187215/139525445-cfb34d9e-0772-4658-ae72-1b303652c77a.png)
