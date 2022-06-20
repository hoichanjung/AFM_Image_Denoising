"""

Copyright (C) 2021 Hoichan JUNG <hoichanjung@korea.ac.kr> - All Rights Reserved

"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class DatasetGenerator(Dataset):
    def __init__(self, pathDirData, dataset, noise_type, transform, logger):

        self.originalDir = pathDirData + '/Original'
        self.noiseDir = pathDirData + f'/{noise_type}'
        self.transform = transform
        self.ImagePaths = []

        imageList = os.listdir(self.originalDir)
        train, test = train_test_split(imageList, test_size=0.2, shuffle=False)
        train, valid = train_test_split(train, test_size=0.25, shuffle=False)

        if dataset == 'train':
            self.ImagePaths = train
        
        if dataset == 'valid':
            self.ImagePaths = valid

        elif dataset == 'test':
            self.ImagePaths = test

        logger.info(f'{dataset} images : {self.ImagePaths}')

    def __getitem__(self, index):

        imagePath = self.ImagePaths[index]

        img_og_name = os.path.join(self.originalDir, imagePath)
        image_og = Image.open(img_og_name)

        img_noise_name = os.path.join(self.noiseDir, imagePath)
        image_noise = Image.open(img_noise_name)
        
        sample = {'original' : image_og, 'noise' : image_noise}

        sample['original'] = self.transform(sample['original'])
        sample['noise'] = self.transform(sample['noise'])

        return sample

    def __len__(self):
        return len(self.ImagePaths)