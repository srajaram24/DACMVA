
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
from DNA_mRNA.TrainMSStripped import TrainMSStrippedClass
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
    

    return options.parse_args()



def runTrainMultiSurv(config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest):
    MultisurvTrainer=TrainMSStrippedClass(config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest)    
    MultisurvTrainer.trainCM()

def fitmRNA(save_dir,max_epochs,missingPercent,dataloaderDir=None,gpu=False,dataLocationTest=None):
    
   
    thisConfig={
        "batch_size" : 64,
        "lrMS" : 1e-5,
        "cancer_type":["BLCA","BRCA","HNSC","KIRC","LGG","LIHC","LUAD","LUSC","OV","STAD"],
        "cancLabel":"All",
        "RNAValSize":32, #use 32 for All or for big subsets, use the size of the entire Val set  for a singel small cancer
        }
    
    
    for bs in [4]:#[8,16,24,32,64]:
        thisConfig["batch_size"]=bs
        runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
                      missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest)
    
  
 
        
  
if __name__ == '__main__':
    args = setup_args()
    if not torch.cuda.is_available():
        args.gpu = -1
    fitmRNA(args.save_dir,args.max_epochs,
         missingPercent=args.missingPercent,dataloaderDir=args.dataloaderDir,gpu=args.gpu,dataLocationTest=args.dataLocationTest)
    




