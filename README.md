# Atomic Force MicroScope(AFM) Image Denoising

## Requirements
- Python 3.7.0
- Cuda 11.1
- pip install -r requirements.txt

## Data Description
Data available at https://drive.google.com/drive/folders/1ChhN62Z_0zL-dqOZt-5UL1vwYGwMM4-5?usp=sharing
- raw_data.zip (raw data)
- 0802_Dataset.zip (ground-truth and input data)


### Filtered Image (by Seung Jun JUNG)
![image](https://user-images.githubusercontent.com/59187215/174540095-6379d112-1a53-428b-a0f7-ba671a4f70d7.png)

### Sample Image (652 AFM Images with Random/Line/Scar/Hum Noise)
![image](https://user-images.githubusercontent.com/59187215/168575051-cc86d871-c79f-46fd-9277-0a78d5b1f904.png)

## Paper
- Comparative Study of Deep Learning Algorithms for Atomic Force Microscope Image Denoising (Hoichan Jung, Giwoong Han, Seong Jun Jung and Sung Won Han)
> Atomic force microscopy (AFM) enables direct visualisation of surface topography at the nanoscale. However, post-processing is generally required to obtain accurate, precise, and reliable AFM images owing to the presence of image artefacts. In this study, we compared and analysed state-of-the-art deep learning models, namely MPRNet, HINet, Uformer, and Restormer, with respect to denoising of AFM images containing four types of noise. Specifically, the denoising performance and inference time of these algorithms on AFM images were compared. Peak signal-to-noise ratio and structural similarity index map were used to evaluate the denoising performance. Based on comparative analysis, Restormer is the most effective and HINet is the most efficient model for denoising AFM images.

## Quantitative Result
### Comparing PSNR and SSIM of Image Denoising Deep Learning Models on AFM Images with Random, Line, Scar, and Hum Noise.
![image](https://user-images.githubusercontent.com/59187215/168575186-0e9086dd-8147-4bfd-9f88-50075ae36cc6.png)
### PSNR vs. inference time on the AFM image
![image](https://user-images.githubusercontent.com/59187215/168575778-1aea355e-925b-4e99-9932-a06e4a9023cc.png)

## Qualitative Result
### Random Noise
![image](https://user-images.githubusercontent.com/59187215/168575270-d9a418aa-1c49-4b88-be8a-dc810abdaba6.png)
### Line Noise
![image](https://user-images.githubusercontent.com/59187215/168575281-d6d866eb-a41e-42af-8805-3c9ed6784829.png)
### Scar Noise
![image](https://user-images.githubusercontent.com/59187215/168575289-86cd9228-0265-4877-91ee-82398dec7911.png)
### Hum Noise
![image](https://user-images.githubusercontent.com/59187215/168575300-14c80cb6-1712-41a2-a7ac-9ff4458ac6a2.png)
