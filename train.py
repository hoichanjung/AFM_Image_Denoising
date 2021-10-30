import time
import logging
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler

from config import getConfig
from DatasetGenerator import DatasetGenerator
from models import * 
from utils import *
from losses import *

cfg = getConfig()

class Trainer():
    def train(pathDirData, noiseType, nnArchitecture, batchSize, trMaxEpoch, imgtransResize, modelName, logger):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'UNET':
            model = UNet(n_classes = 1, depth = cfg.depth, padding = True).cuda()
            criterion = nn.MSELoss()

        elif nnArchitecture == 'HINET':
            model = HINet(in_chn = 1).cuda()
            criterion_char = CharbonnierLoss()
            criterion_edge = EdgeLoss()
            criterion_psnr = PSNRLoss()

        model = torch.nn.DataParallel(model).cuda()    
        pathModel = f'results/{modelName}/model.pth.tar'

        # -------------------- SETTINGS: DATA TRANSFORMS
        # normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        # transformList.append(transforms.RandomHorizontalFlip(p=0.5))
        # transformList.append(transforms.RandomVerticalFlip(p=0.5))
        # transformList.append(transforms.RandomRotation(degrees=180))
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.ToTensor())
        transform = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathDirData, dataset='train', noise_type=noiseType, transform=transform, logger=logger)
        datasetValid = DatasetGenerator(pathDirData, dataset='valid', noise_type=noiseType, transform=transform, logger=logger)

        dataLoaderTrain = DataLoader(dataset = datasetTrain, batch_size = batchSize , shuffle = True, pin_memory = True)
        dataLoaderValid = DataLoader(dataset = datasetValid, batch_size = batchSize , shuffle = False, pin_memory = True)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        if cfg.op == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=cfg.patience, mode='min')
        elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
            warmup_epochs = 3
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, trMaxEpoch-warmup_epochs, eta_min=1e-8)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

        # -------------------- TRAIN / VALIDATION
        lossMIN = 10000

        for epochID in range(trMaxEpoch):
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            print('current lr : {:.7f}'.format(optimizer.param_groups[0]['lr']))

            if nnArchitecture == 'UNET':
                epochTrainUNET(model, dataLoaderTrain, optimizer, criterion)
                lossVal, losstensor = epochValidUNET(model, dataLoaderValid, criterion, epochID, modelName)

            if nnArchitecture == 'HINET':
                # epochTrainHINET(model, dataLoaderTrain, optimizer, criterion_char, criterion_edge)
                epochTrainHINET(model, dataLoaderTrain, optimizer, criterion_char, criterion_edge, criterion_psnr)
                lossVal, losstensor = epochValidHINET(model, dataLoaderValid, criterion_char, criterion_edge, criterion_psnr, epochID, modelName)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(losstensor.data)

            if lossVal < lossMIN:
                lossMIN = lossVal

                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()}, pathModel)
                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
                logger.info('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                logger.info('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))

    def test(pathDirData, noiseType, nnArchitecture, batchSize, imgtransResize, modelName, logger):
        cudnn.benchmark = True
        
        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'UNET':
            model = UNet(n_classes = 1, depth = cfg.depth, padding = True).cuda()
            
        elif nnArchitecture == 'HINET':
            model = HINet(in_chn = 1).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
        pathModel = f'results/{modelName}/model.pth.tar'
        load_pretrained(model, pathModel)

        # -------------------- SETTINGS: DATA TRANSFORMS
        # normalize = transforms.Normalize([0.5], [0.5])
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.ToTensor())
        transform = transforms.Compose(transformList)
        
        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTest = DatasetGenerator(pathDirData, dataset = 'test', noise_type = noiseType, transform = transform, logger=logger)
        dataLoaderTest = DataLoader(dataset = datasetTest, batch_size = batchSize , shuffle = False, pin_memory = True)

        # -------------------- TEST
        with torch.no_grad():
            
            model.eval()

            label = torch.FloatTensor().cuda()
            noise = torch.FloatTensor().cuda()
            pred = torch.FloatTensor().cuda()
            
            imgNumber = 0
            for batchID, data in enumerate(tqdm(dataLoaderTest)):
            
                img_og, img_noise = data['original'].cuda(), data['noise'].cuda()
                img_output = model(img_noise)

                label = torch.cat((label, img_og), 0)
                noise = torch.cat((noise, img_noise), 0)        
                
                if nnArchitecture == 'UNET':
                    pred = torch.cat((pred, img_output), 0)
                elif nnArchitecture == 'HINET':
                    pred = torch.cat((pred, img_output[1]), 0)

                for imgID in range(img_og.shape[0]):
                    
                    save_image(img_og[imgID], f'./results/{modelName}/original/{imgNumber}.png')
                    save_image(img_noise[imgID], f'./results/{modelName}/noise/{imgNumber}.png')
                    
                    if nnArchitecture == 'UNET':
                        save_image(img_output[imgID], f'./results/{modelName}/output/{imgNumber}.png')
                    elif nnArchitecture == 'HINET':
                        save_image(img_output[1][imgID], f'./results/{modelName}/output/{imgNumber}.png')
                    
                    imgNumber += 1

            noisePSNR = torchPSNR(label, noise)
            predPSNR = torchPSNR(label, pred)
            print(f"Noise PSNR : {noisePSNR: .2f}")
            print(f"Pred PSNR : {predPSNR: .2f}")
            
            logger.info(f"Noise PSNR : {noisePSNR: .2f}")
            logger.info(f"Pred PSNR : {predPSNR: .2f}")
