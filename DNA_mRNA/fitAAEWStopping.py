#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:31:02 2023

@author: sara
"""
import torch
import torch.utils.data

import sys

import os
sys.path.append(os.getcwd())
#print(sys.path)

import argparse

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from DNA_mRNA.TrainAAEWStopping import TrainAAEImputeWStoppingClass
from functools import partial


torch.manual_seed(42)

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=401, type=int)
    options.add_argument('-dataloaderDir', '--dataloaderDir', action="store", dest="dataloaderDir",default=None)
    options.add_argument('-mp', '--missingPercent', action="store", dest="missingPercent", default=0.0, type=float)

    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
    options.add_argument('-dataLocationTest', '--dataLocationTest', action="store", dest="dataLocationTest",default=None)
    
    options.add_argument('-modelLocation', '--modelLocation', action="store", dest="modelLocation",default=None)
    

    return options.parse_args()


def runTrainMultiSurv(config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,modelLocation):
    MultisurvTrainer=TrainAAEImputeWStoppingClass(config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,modelLocation,fixedConfig=True)    
    MultisurvTrainer.trainCM()

def fitDNAmRNA(save_dir,max_epochs,missingPercent,dataloaderDir=None,gpu=False,dataLocationTest=None,modelLocation=None):
    
   
        
    thisConfig={
        "lrAE": 0.0001,
        "lrD": 5e-5,
        "beta": 0.0,
        "batch_size" : 16,
        "latent_dims" : 512,
        "alpha" : 1.0,
  
        }
    
  
   
    thisConfig["batch_size"]=16
    for b in [0.0,0.2,0.6,0.8]:
        thisConfig["beta"]=b
        runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
                          missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation)
    
    thisConfig["batch_size"]=32
    for b in [0.0,0.2,0.6,0.8]:
        thisConfig["beta"]=b
        runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
                          missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation)
            
  

if __name__ == '__main__':
    args = setup_args()
    if not torch.cuda.is_available():
        args.gpu = -1
    fitDNAmRNA(args.save_dir,args.max_epochs,
         missingPercent=args.missingPercent,dataloaderDir=args.dataloaderDir,gpu=args.gpu,dataLocationTest=args.dataLocationTest,modelLocation=args.modelLocation)
    




