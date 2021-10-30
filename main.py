import os
import time
import pprint
import pandas as pd
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
    

    if cfg.action == 'train':
        runTrain()
    elif cfg.action == 'test':
        runTest()

#--------------------------------------------------------------------------------
def runTrain():

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
    Trainer.train(pathDirData, noiseType, nnArchitecture, batchSize, trMaxEpoch, imgtransResize, modelName, logger)

    # -------------------- TEST    
    print('Testing the trained model')
    Trainer.test(pathDirData, noiseType, nnArchitecture, batchSize, imgtransResize, modelName, logger)

#--------------------------------------------------------------------------------
def runTest():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gid
    
    pathDirData = '../0802_Dataset/'
    expNum = cfg.exp_num
    noiseType = cfg.noise
    nnArchitecture = cfg.model

    imgtransResize = cfg.resize
    batchSize = cfg.b
    
    modelName = f'Denoising_AFM_{nnArchitecture}_{noiseType}_exp{expNum}_{time.strftime("%m%d")}'
    
    make_results_directory(modelName)
    logger = get_model_logger(modelName)
    # -------------------- TEST
    print('Testing the trained model')
    Trainer.test(pathDirData, noiseType, nnArchitecture, batchSize, imgtransResize, modelName, logger)

#--------------------------------------------------------------------------------

if __name__ == '__main__': 
    main()
