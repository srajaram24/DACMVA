
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


from DNA_mRNA.TrainMSStripped import TrainMSStrippedClass

import pickle

torch.manual_seed(42)

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=401, type=int)
    options.add_argument('-dataloaderDirImputed', '--dataloaderDirImputed', action="store", dest="dataloaderDirImputed",default=None)
    options.add_argument('-dataloaderDirVal', '--dataloaderDirVal', action="store", dest="dataloaderDirVal",default=None)
    options.add_argument('-mp', '--missingPercent', action="store", dest="missingPercent", default=0.0, type=float)

    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
    options.add_argument('-dataLocationTest', '--dataLocationTest', action="store", dest="dataLocationTest",default=None)
    

    return options.parse_args()



def runTrainMultiSurv(config,save_dir,max_epochs,missingPercent,gpu,dataLocationTest,dlTrain,dlVal):
    MultisurvTrainer=TrainMSStrippedClass(config,save_dir,max_epochs,missingPercent,None,gpu,dataLocationTest,dlTrain,dlVal)    
    MultisurvTrainer.trainCM()

def fitmRNA(save_dir,max_epochs,missingPercent,dataloaderDirImputed,dataloaderDirVal,gpu=False,dataLocationTest=None):
    
   
    thisConfig={
        "batch_size" : 64,
        "lrMS" : 1e-5,
        "cancer_type":["BLCA","BRCA","HNSC","KIRC","LGG","LIHC","LUAD","LUSC","OV","STAD"],
        "cancLabel":"All",
        "RNAValSize":25, #use 32 for All or for big subsets, use 25 for mp95, 55 for mp90
        }
    

    for bs in [4,8,16,24,32,64]:
        thisConfig["batch_size"]=bs
        with open(os.path.join(dataloaderDirImputed,"dataloaderAllImputedTrain"+str(bs)+"mp"+str(int(missingPercent*100))), 'rb') as file:
             dlTrain = pickle.load(file)
        
        with open(os.path.join(dataloaderDirVal,"dataloader"+thisConfig["cancLabel"]+"RNAVal"+str(thisConfig["RNAValSize"])+"mp"+str(int(missingPercent*100))), 'rb') as file:
             dlVal = pickle.load(file)
             
        runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
                      missingPercent=missingPercent, gpu=gpu,dataLocationTest=dataLocationTest,dlTrain=dlTrain, dlVal=dlVal)
    
  
 
        
  
if __name__ == '__main__':
    args = setup_args()
    if not torch.cuda.is_available():
        args.gpu = -1
    fitmRNA(args.save_dir,args.max_epochs,
         missingPercent=args.missingPercent,dataloaderDirImputed=args.dataloaderDirImputed,dataloaderDirVal=args.dataloaderDirVal,gpu=args.gpu,dataLocationTest=args.dataLocationTest)
    




