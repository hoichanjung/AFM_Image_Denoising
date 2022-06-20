"""

Copyright (C) 2021 Hoichan JUNG <hoichanjung@korea.ac.kr> - All Rights Reserved

"""

import os
import time
import pprint
import pandas as pd
import neptune.new as neptune
import warnings
warnings.filterwarnings(action='ignore')

from train import Trainer
from config import getConfig
from utils import *

random_seed = 777
manage_reproducibility(random_seed)

cfg = getConfig()

def main():

    print('<---- Training Params ---->')
    pprint.pprint(cfg)

    run = neptune.init(project="fdai.hoichan.d.jung/AFMdenoising",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODU1NzNkNi03MWE4LTRhZjktOTM2Yi03OGFhMThiY2E2ODMifQ==",
                    mode=cfg.neptune)
    
    run['model'] = cfg.model
    run['noise'] = cfg.noise
    run['exp_num'] = cfg.exp_num
    run['epoch'] = cfg.epochs

    if cfg.action == 'train':
        runTrain(run)
    elif cfg.action == 'test':
        runTest(run)

#--------------------------------------------------------------------------------
def runTrain(run):

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gid

    pathDirData = '../0802_Dataset/'
    expNum = cfg.exp_num
    noiseType = cfg.noise
    nnArchitecture = cfg.model

    imgtransResize = cfg.resize
    batchSize = cfg.b
    trMaxEpoch = cfg.epochs
    
    modelName = f'Denoising_AFM_{nnArchitecture}_{noiseType}_exp{expNum}_{time.strftime("%m%d")}'
    
    make_results_directory(modelName)
    logger = get_model_logger(modelName)
    # -------------------- TRAIN / VALIDATION
    print('Training NN architecture = ', nnArchitecture)
    Trainer.train(pathDirData, noiseType, nnArchitecture, batchSize, trMaxEpoch, imgtransResize, modelName, logger, run)

    # -------------------- TEST    
    print('Testing the trained model')
    Trainer.test(pathDirData, noiseType, nnArchitecture, batchSize, imgtransResize, modelName, logger, run)

#--------------------------------------------------------------------------------
def runTest(run):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gid
    
    pathDirData = '../0802_Dataset/'
    expNum = cfg.exp_num
    noiseType = cfg.noise
    nnArchitecture = cfg.model

    imgtransResize = cfg.resize
    batchSize = cfg.b
    
    # modelName = f'Denoising_AFM_{nnArchitecture}_{noiseType}_exp{expNum}_{time.strftime("%m%d")}'
    modelName = cfg.ckpt
    
    make_results_directory(modelName)
    logger = get_model_logger(modelName)
    # -------------------- TEST
    print('Testing the trained model')
    Trainer.test(pathDirData, noiseType, nnArchitecture, batchSize, imgtransResize, modelName, logger, run)

#--------------------------------------------------------------------------------

if __name__ == '__main__': 
    main()
