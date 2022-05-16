"""

Copyright (C) 2021 Hoichan JUNG <hoichanjung@korea.ac.kr> - All Rights Reserved

"""

import time
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
from IQA_pytorch import SSIM
from ptflops import get_model_complexity_info

from config import getConfig
from DatasetGenerator import DatasetGenerator
from models import * 
from utils import *
from losses import *

cfg = getConfig()

class Trainer():
    def train(pathDirData, noiseType, nnArchitecture, batchSize, trMaxEpoch, imgtransResize, model_name, logger, run):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'UNET':
            model = Denoising_UNet(in_channels=1, n_classes=1, padding=True).cuda()
        elif nnArchitecture == 'REDNET':
            model = REDNet20(in_channels=1, num_layers=6).cuda()            
        elif nnArchitecture == 'UNET_REDNET':
            model = UNet_REDNet().cuda()
        elif nnArchitecture == 'VDSR':
            model = VDSR().cuda()            
        elif nnArchitecture == 'HINET':
            model = HINet(in_chn = 1).cuda()
        elif nnArchitecture == 'MPRNET':
            model = MPRNet(in_c = 1, out_c=1).cuda()      
        elif nnArchitecture == 'UFORMER':
            model = Uformer(img_size=imgtransResize, in_chans=1).cuda()
        elif nnArchitecture == 'RESTORMER':
            model = Restormer(inp_channels=1, out_channels=1, LayerNorm_type='BiasFree').cuda()     
        elif nnArchitecture == 'NAFNET':
            model = NAFNet(img_channel=1).cuda()     

        macs, params = get_model_complexity_info(model, (1, imgtransResize, imgtransResize), as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
        print('MACs(G): ', macs)
        print('Params: ', params)
        
        logger.info(f'MACs(G): {macs}')
        logger.info(f'Params: {params}')

        run['MACs(G)'] = macs
        run['Params'] = params 

        model = torch.nn.DataParallel(model).cuda()    
        pathModel = f'results/{model_name}/model.pth.tar'
       
        # -------------------- SETTINGS: DATA TRANSFORMS
        # normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        # transformList.append(transforms.RandomHorizontalFlip())
        # transformList.append(transforms.RandomVerticalFlip())
        # transformList.append(transforms.RandomAffine((-20, 20)))
        # transformList.append(transforms.RandomRotation(90))
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.ToTensor())
        transform = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathDirData, dataset='train', noise_type=noiseType, transform=transform, logger=logger)
        datasetValid = DatasetGenerator(pathDirData, dataset='valid', noise_type=noiseType, transform=transform, logger=logger)

        dataLoaderTrain = DataLoader(dataset = datasetTrain, batch_size = batchSize , shuffle = True, pin_memory = True)
        dataLoaderValid = DataLoader(dataset = datasetValid, batch_size = batchSize , shuffle = False, pin_memory = True)
        run['Batch'] = batchSize
        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        if cfg.op == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=cfg.patience, mode='min')
        elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
            warmup_epochs = 3
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, trMaxEpoch-warmup_epochs, eta_min=1e-8)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        
        # -------------------- SETTINGS: Loss Function
        # criterion_mse = nn.MSELoss()
        criterion_char = CharbonnierLoss()
        criterion_edge = EdgeLoss()         
        # criterion_psnr = PSNRLoss()

        # -------------------- TRAIN / VALIDATION
        lossMIN = 10000

        for epochID in range(trMaxEpoch):
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            print('current lr : {:.7f}'.format(optimizer.param_groups[0]['lr']))

            if nnArchitecture == 'UNET' or nnArchitecture == 'REDNET' or nnArchitecture == 'UNET_REDNET'\
                or nnArchitecture == 'VDSR' or nnArchitecture == 'UFORMER' or nnArchitecture == 'RESTORMER'\
                or nnArchitecture == 'NAFNET':
                epochTrainUNET(model, dataLoaderTrain, optimizer, criterion_char, criterion_edge)
                lossVal, losstensor = epochValidUNET(model, dataLoaderValid, criterion_char, criterion_edge, epochID, model_name)

            elif nnArchitecture == 'HINET' or nnArchitecture == 'MPRNET':
                epochTrainHINET(model, dataLoaderTrain, optimizer, criterion_char, criterion_edge)
                lossVal, losstensor = epochValidHINET(model, dataLoaderValid, criterion_char, criterion_edge, epochID, model_name)

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
                run['Valid loss'].log(lossVal)

            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                logger.info('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))

    def test(pathDirData, noiseType, nnArchitecture, batchSize, imgtransResize, model_name, logger, run):
        cudnn.benchmark = True
        
        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'UNET':
            model = Denoising_UNet(in_channels=1, n_classes=1, padding=True).cuda()
        elif nnArchitecture == 'REDNET':
            model = REDNet20(in_channels=1, num_layers=6).cuda()
        elif nnArchitecture == 'UNET_REDNET':
            model = UNet_REDNet().cuda()                      
        elif nnArchitecture == 'VDSR':
            model = VDSR().cuda()  
        elif nnArchitecture == 'HINET':
            model = HINet(in_chn = 1).cuda()
        elif nnArchitecture == 'MPRNET':
            model = MPRNet(in_c = 1, out_c=1).cuda()      
        elif nnArchitecture == 'UFORMER':
            model = Uformer(img_size=imgtransResize, in_chans=1).cuda()
        elif nnArchitecture == 'RESTORMER':
            model = Restormer(inp_channels=1, out_channels=1, LayerNorm_type='BiasFree').cuda()      
        elif nnArchitecture == 'NAFNET':
            model = NAFNet(img_channel=1).cuda() 

        model = torch.nn.DataParallel(model).cuda()
        pathModel = f'results/{model_name}/model.pth.tar'
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
            
            img_number = 0
            for batchID, data in enumerate(tqdm(dataLoaderTest)):
            
                img_og, img_noise = data['original'].cuda(), data['noise'].cuda()
                img_output = model(img_noise)

                label = torch.cat((label, img_og), 0)
                noise = torch.cat((noise, img_noise), 0)        
                
                if nnArchitecture == 'UNET' or nnArchitecture == 'REDNET' or nnArchitecture == 'UNET_REDNET'\
                    or nnArchitecture == 'VDSR' or nnArchitecture == 'UFORMER' or nnArchitecture == 'RESTORMER'\
                    or nnArchitecture == 'NAFNET':
                    pred = torch.cat((pred, img_output), 0)
                elif nnArchitecture == 'HINET':
                    pred = torch.cat((pred, img_output[1]), 0)
                elif nnArchitecture == 'MPRNET':
                    pred = torch.cat((pred, img_output[0]), 0)

                for imgID in range(img_og.shape[0]):
                    
                    save_image(img_og[imgID], f'./results/{model_name}/original/{img_number}.png')
                    save_image(img_noise[imgID], f'./results/{model_name}/noise/{img_number}.png')
                    
                    if nnArchitecture == 'UNET' or nnArchitecture == 'REDNET' or nnArchitecture == 'UNET_REDNET'\
                        or nnArchitecture == 'VDSR' or nnArchitecture == 'UFORMER' or nnArchitecture == 'RESTORMER':
                        save_image(img_output[imgID], f'./results/{model_name}/output/{img_number}.png')
                    elif nnArchitecture == 'HINET':
                        save_image(img_output[1][imgID], f'./results/{model_name}/output/{img_number}.png')
                    elif nnArchitecture == 'MPRNET':
                        save_image(img_output[0][imgID], f'./results/{model_name}/output/{img_number}.png')
                    
                    img_number += 1

            noisePSNR = torchPSNR(label, noise)
            predPSNR = torchPSNR(label, pred)
            print(f"Noise PSNR : {noisePSNR: .2f}")
            print(f"Pred PSNR : {predPSNR: .2f}")
            
            logger.info(f"Noise PSNR : {noisePSNR: .2f}")
            logger.info(f"Pred PSNR : {predPSNR: .2f}")

            run['Noise PSNR'] = f"{noisePSNR: .2f}" 
            run['Pred PSNR'] = f"{predPSNR: .2f}"

            ssim = SSIM()
            noiseSSIM = torch.mean(ssim(label, noise, as_loss=False))
            predSSIM = torch.mean(ssim(label, pred, as_loss=False))       
            
            print(f"Noise SSIM : {noiseSSIM: .3f}")
            print(f"Pred SSIM : {predSSIM: .3f}")
            
            logger.info(f"Noise SSIM : {noiseSSIM: .3f}")
            logger.info(f"Pred SSIM : {predSSIM: .3f}")

            run['Noise SSIM'] = f"{noiseSSIM: .3f}" 
            run['Pred SSIM'] = f"{predSSIM: .3f}"

            save_results(label, noise, pred, f'./results/{model_name}/results.csv')
            
       
            
            