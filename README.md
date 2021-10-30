# Atomic Force MicroScope(AFM) Image Denoising

## Requirements
- Python 3.8.11
- Cuda 11.1
- pip install -r requirements.txt

## Data Description
1. 652 Dummy AFM Images with 3 Types of Noise (Random/Line/Scar)
2. AFM Image collected from SKHynix (by Seung Jun JUNG)

## Model
- HINet: Half Instance Normalization Network for Image Restoration
- Paper: https://arxiv.org/abs/2105.06086

### Network Architecture
![image](https://user-images.githubusercontent.com/59187215/139525445-cfb34d9e-0772-4658-ae72-1b303652c77a.png)
