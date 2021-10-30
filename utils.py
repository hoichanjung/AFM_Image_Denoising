import os
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

# -------------------- MODEL TRAIN
def epochTrainUNET(model, dataLoader, optimizer, criterion):
    model.train()
        
    for batchID, data in enumerate(tqdm(dataLoader)):
        img_og, img_noise = data['original'].cuda(), data['noise'].cuda()
        
        img_output = model(img_noise.float())
        loss = criterion(img_output.float(), img_og.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def epochValidUNET(model, dataLoader, criterion, epochID, modelName):
    
    with torch.no_grad():
        model.eval()
        
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0

        for batchID, data in enumerate(tqdm(dataLoader)):
            img_og, img_noise = data['original'].cuda(), data['noise'].cuda()
            img_output = model(img_noise.float())
            loss = criterion(img_output.float(), img_og.float())

            # -------------------- SAVE SAMPLE IMAGE
            if batchID == 0:
                save_image(img_output[0], f'./results/{modelName}/sample/sample_epoch{epochID}.png')

            losstensorMean += loss
            lossVal += loss
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean

def epochTrainHINET(model, dataLoader, optimizer, criterion_char, criterion_edge, criterion_psnr):
    model.train()
        
    for batchID, data in enumerate(tqdm(dataLoader)):
        img_og, img_noise = data['original'].cuda(), data['noise'].cuda()
        img_output = model(img_noise)

        # loss_char = sum([criterion_char(img_output[j], img_og) for j in range(len(img_output))])  # Charbonnier Loss
        # loss_edge = sum([criterion_edge(img_output[j], img_og) for j in range(len(img_output))])  # Edge Loss               
        # loss = (loss_char) + (0.05*loss_edge)    
        
        # loss_char = sum([criterion_char(img_output[j], img_og) for j in range(len(img_output))])  # Charbonnier Loss
        # loss = loss_char

        loss_psnr = sum([criterion_psnr(img_output[j], img_og) for j in range(len(img_output))])  # PSNR Loss
        loss = loss_psnr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def epochValidHINET(model, dataLoader, criterion_char, criterion_edge, criterion_psnr, epochID, modelName):
    
    with torch.no_grad():
        model.eval()
        
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0

        for batchID, data in enumerate(tqdm(dataLoader)):
            img_og, img_noise = data['original'].cuda(), data['noise'].cuda()
            img_output = model(img_noise)

            # loss_char = sum([criterion_char(img_output[j], img_og) for j in range(len(img_output))])  # Charbonnier Loss
            # loss_edge = sum([criterion_edge(img_output[j], img_og) for j in range(len(img_output))])  # Edge Loss
            # loss = (loss_char) + (0.05*loss_edge)    
        
            # loss_char = sum([criterion_char(img_output[j], img_og) for j in range(len(img_output))])  # Charbonnier Loss
            # loss = loss_char
            
            loss_psnr = sum([criterion_psnr(img_output[j], img_og) for j in range(len(img_output))])  # PSNR Loss
            loss = loss_psnr

            # -------------------- SAVE SAMPLE IMAGE
            if batchID == 0:
                save_image(img_output[1][0], f'./results/{modelName}/sample/sample_epoch{epochID+1}.png')

            losstensorMean += loss
            lossVal += loss
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean

def load_pretrained(model, pathModel):
    modelCheckpoint = torch.load(pathModel)
    state_dict = modelCheckpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    print('###### pre-trained Model restored #####')

# -------------------- EVALUATION METRICS
def MSE(y_true, y_pred):
    y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    return np.square(np.subtract(y_true, y_pred)).mean()

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

# -------------------- OTHERS
def manage_reproducibility(random_seed): 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def make_results_directory(modelName):
    os.makedirs('results/', exist_ok=True)
    os.makedirs(f'results/{modelName}/', exist_ok=True)
    os.makedirs(f'results/{modelName}/sample/', exist_ok=True)
    os.makedirs(f'results/{modelName}/original/', exist_ok=True)
    os.makedirs(f'results/{modelName}/noise/', exist_ok=True)
    os.makedirs(f'results/{modelName}/output/', exist_ok=True)

def get_model_logger(modelName):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)s] %(message)s')
    file_handler = logging.FileHandler(f'results/{modelName}/log.log', "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger