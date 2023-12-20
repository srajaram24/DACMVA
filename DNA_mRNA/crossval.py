#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:46:57 2023

@author: sara
"""



import torch
import torch.utils.data

import sys

import os
sys.path.append(os.getcwd())
#print(sys.path)

import argparse
from sklearn.model_selection import KFold

from DNA_mRNA.TrainmRNAImpute import TrainmRNAImputeClass
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from DNA_mRNA.TrainOversample import TrainOversamplingClass
from DNA_mRNA.TrainMSStripped import TrainMSStrippedClass

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=350, type=int)
    options.add_argument('-dataloaderDir', '--dataloaderDir', action="store", dest="dataloaderDir",default=None)
    options.add_argument('-mp', '--missingPercent', action="store", dest="missingPercent", default=0.0, type=float)

    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
    options.add_argument('-dataLocationTest', '--dataLocationTest', action="store", dest="dataLocationTest",default=None)
    


    


    return options.parse_args()

if __name__ == '__main__':
    args = setup_args()
    if not torch.cuda.is_available():
        args.gpu = -1
        
        
    with open(os.path.join(args.dataloaderDir,"dataloaderAll"+"RNAVal"+str(32)+"mp"+str(int(args.missingPercent*100))), 'rb') as file:
         dataloaderRNAValCanc = pickle.load(file)
    
    with open(os.path.join(args.dataloaderDir,"dataloaderAll"+"RNATrain"+str(32)+"mp"+str(int(args.missingPercent*100))), 'rb') as file:
         dataloaderRNATrainCanc = pickle.load(file)
    
    with open(os.path.join(args.dataloaderDir,"dataloaderAll"+"RNATest"+str(32)+"mp"+str(int(args.missingPercent*100))), 'rb') as file:
         dataloaderRNATestCanc = pickle.load(file)
    
    datasetTrain=dataloaderRNATrainCanc.dataset
    datasetVal=dataloaderRNAValCanc.dataset
    datasetTest=dataloaderRNATestCanc.dataset
    

    trainValSet = ConcatDataset([datasetTrain, datasetVal,datasetTest])

    print("Len of trainValSet is "+str(len(trainValSet)))
    configBase={'batch_size': 24, 'lrMS': 1e-05, 'cancer_type': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'], 'cancLabel': 'All', 'RNAValSize': 55}
    
    
    configOV={'batch_size': 8, 'gamma': 0.1, 'lrMS': 1e-05, 'cancer_type': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'], 'cancLabel': 'All', 'bs_Oversample': 24, 'RNAValSize': 55}
    
    
    configAEAdv={
        "batch_size" : 8,
        "latent_dims" : 32,
        "gamma" : 0.6,
        "lrMS" : 1e-5,
        "t" : 0.1,
        "cancer_type":["BLCA","BRCA","HNSC","KIRC","LGG","LIHC","LUAD","LUSC","OV","STAD"],
        "cancLabel":"All",
        "RNAValSize":55, #use 32 for All or for big subsets, use the size of the entire Val set  for a singel small cancer
        "ImputeBS":32,
        "UnpairedDNA":False,
        "tag":""
        }
    modelLocationAEAdv='/nethome/srajaram8/TrainDNAmRNAmp90/TrainAECox/AAE/AdvCM/lrAE_0.0001_lrD_5e-05_VAE_True_bs_32_lrAE_0.0001_ld_32_numLayersA_3_numLayersB_3_lambB_0.0001_cm_0.6_anc_0.0_FixedAdv100Delay_CMLoss/modelfiles/'
    
    
    configAENoAdv={'batch_size': 8, 'latent_dims': 32, 'gamma': 0.6, 'lrMS': 1e-05, 't': 0.1, 'cancer_type': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 
'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'], 'cancLabel': 'All', 'RNAValSize': 55, 'ImputeBS': 32, 'UnpairedDNA': False, 'tag': ''}
    modelLocationAENoAdv='/nethome/srajaram8/TrainDNAmRNAmp90/TrainAECox/AAE/NoAdvCM/lrAE_0.0001_lrD_5e-05_VAE_True_bs_32_lrAE_0.0001_ld_32_numLayersA_3_numLayersB_3_lambB_0.0001_cm_0.6_anc_0.0_FixedNoAdv_CMLoss/modelfiles'
    
    configMSTDImpute=
    
    k=10
    
   
   
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
        
        print("k="+str(k), file=f)
        print(configBase, file=f)
        print(configOV, file=f)
        print(modelLocationAEAdv, file=f)
        print(configAEAdv, file=f)
        print("missing percent is "+str(args.missingPercent), file=f)
        print("Len of trainValSet is "+str(len(trainValSet)), file=f)
    
    
    os.makedirs(os.path.join(args.save_dir,"kfoldsSets"), exist_ok=True)
    
    
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    CtD1=np.array([])
    CtD2=np.array([])
    CtD3=np.array([])
    CtD4=np.array([])
    CtD5=np.array([])
    
    IBS1=np.array([])
    IBS2=np.array([])
    IBS3=np.array([])
    IBS4=np.array([])
    IBS5=np.array([])
    
    saveDir=os.path.join(args.save_dir,"Base")
    os.makedirs(saveDir, exist_ok=True)
    saveDir=os.path.join(args.save_dir,"OV")
    os.makedirs(saveDir, exist_ok=True)
    saveDir=os.path.join(args.save_dir,"AEAdv")
    os.makedirs(saveDir, exist_ok=True)
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(trainValSet)))):

        print('####################Fold {}###############################3'.format(fold + 1))
        
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            print('Fold {}'.format(fold + 1), file=f)
            
            
        
    
        trainSet=[trainValSet[i] for i in train_idx]
        valSet=[trainValSet[i] for i in val_idx]
        
        
        fileName="trainSet"+str(fold+1)
        with open(os.path.join(args.save_dir,"kfoldsSets",fileName),'wb') as file:
            pickle.dump(trainSet,file)
        
        fileName="valSet"+str(fold+1)
        with open(os.path.join(args.save_dir,"kfoldsSets",fileName),'wb') as file:
            pickle.dump(valSet,file)
        
        if len(val_idx)<129:
            RNAValSize=len(val_idx)
        else:
            RNAValSize=configBase["RNAValSize"]
        
        
        train_loaderBase = DataLoader(trainSet, batch_size=configBase["batch_size"],shuffle=True,drop_last=True)
        train_loaderOV = DataLoader(trainSet, batch_size=configOV["batch_size"],shuffle=True,drop_last=True)
        train_loaderAEAdv = DataLoader(trainSet, batch_size=configAEAdv["batch_size"],shuffle=True,drop_last=True)
        train_loaderAENoAdv = DataLoader(trainSet, batch_size=configAENoAdv["batch_size"],shuffle=True,drop_last=True)
        train_loaderTDImpute = DataLoader(trainSet, batch_size=configMSTDImpute["batch_size"],shuffle=True,drop_last=True)
        val_loader = DataLoader(valSet, batch_size=RNAValSize, shuffle=True, drop_last=False)

        print("Len of train loader Base for fold is "+str(len(train_loaderBase.dataset)))
        print("Len of train loader OV for fold is "+str(len(train_loaderOV.dataset)))
        print("Len of train loader AEAdv for fold is "+str(len(train_loaderAEAdv.dataset)))
        print("Len of val loader for fold is "+str(len(val_loader.dataset)))
        print("Len of train idx for fold is "+str(len(train_idx)))
        print("Len of val idx for fold is "+str(len(val_idx)))
        
    
         
        saveDir=os.path.join(args.save_dir,"Base","k"+str(fold + 1))
        os.makedirs(saveDir, exist_ok=True)
        MultisurvTrainer1=TrainMSStrippedClass(configBase,saveDir,args.max_epochs,args.missingPercent,None,args.gpu,args.dataLocationTest,train_loaderBase,val_loader)    
        ctdVal,ibsVal=MultisurvTrainer1.trainCM()
        CtD1=np.append(CtD1,ctdVal)
        IBS1=np.append(IBS1,ibsVal)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            print('Baseline Ctd, '+str(ctdVal), file=f)
            print('Baseline IBS, '+str(ibsVal), file=f)
        
        saveDir=os.path.join(args.save_dir,"OV","k"+str(fold + 1))
        os.makedirs(saveDir, exist_ok=True)
        MultisurvTrainer2=TrainOversamplingClass(configOV,saveDir,args.max_epochs,args.missingPercent,None,args.gpu,args.dataLocationTest,train_loaderOV,val_loader)    
        ctdVal,ibsVal=MultisurvTrainer2.trainCM()
        CtD2=np.append(CtD2,ctdVal)
        IBS2=np.append(IBS2,ibsVal)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            print('Oversample Ctd, '+str(ctdVal), file=f)
            print('Oversample IBS, '+str(ibsVal), file=f)
        
        saveDir=os.path.join(args.save_dir,"AEAdv","k"+str(fold + 1))
        os.makedirs(saveDir, exist_ok=True)
        MultisurvTrainer3=TrainmRNAImputeClass(configAEAdv,saveDir,args.max_epochs,args.missingPercent,args.dataloaderDir,args.gpu,args.dataLocationTest,modelLocationAEAdv,True,train_loaderAEAdv,val_loader)
        ctdVal,ibsVal=MultisurvTrainer3.trainCM()
        CtD3=np.append(CtD3,ctdVal)
        IBS3=np.append(IBS3,ibsVal)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            print('AEAdv Model Ctd, '+str(ctdVal), file=f)
            print('AEAdv Model IBS, '+str(ibsVal), file=f)
            
        saveDir=os.path.join(args.save_dir,"AENoAdv","k"+str(fold + 1))
        os.makedirs(saveDir, exist_ok=True)
        MultisurvTrainer4=TrainmRNAImputeClass(configAENoAdv,saveDir,args.max_epochs,args.missingPercent,args.dataloaderDir,args.gpu,args.dataLocationTest,modelLocationAENoAdv,True,train_loaderAENoAdv,val_loader)
        ctdVal,ibsVal=MultisurvTrainer4.trainCM()
        CtD4=np.append(CtD4,ctdVal)
        IBS4=np.append(IBS4,ibsVal)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            print('AENoAdv Model Ctd, '+str(ctdVal), file=f)
            print('AENoAdv Model IBS, '+str(ibsVal), file=f)
            
            
        saveDir=os.path.join(args.save_dir,"MSTDImpute","k"+str(fold + 1))
        os.makedirs(saveDir, exist_ok=True)
        MultisurvTrainer=TrainMSStrippedClass(configMSTDImpute,saveDir,args.max_epochs,args.missingPercent,None,args.gpu,args.dataLocationTest,train_loaderTDImpute,val_loader)
        ctdVal,ibsVal=MultisurvTrainer.trainCM()
        CtD5=np.append(CtD5,ctdVal)
        IBS5=np.append(IBS5,ibsVal)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            print('MSTDImpute Model Ctd, '+str(ctdVal), file=f)
            print('MSTDImpute Model IBS, '+str(ibsVal), file=f)
    
    
    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        
        print('Baseline Ctd Avg, '+str(np.mean(CtD1)), file=f)
        print('Baseline Ctd StdDev, '+str(np.std(CtD1)), file=f)
        print('Baseline IBS Avg, '+str(np.mean(IBS1)), file=f)
        print('Baseline IBS StdDev, '+str(np.std(IBS1)), file=f)
        
        
        print('Oversample Ctd Avg, '+str(np.mean(CtD2)), file=f)
        print('Oversample Ctd StdDev, '+str(np.std(CtD2)), file=f)
        print('Oversample IBS Avg, '+str(np.mean(IBS2)), file=f)
        print('Oversample IBS StdDev, '+str(np.std(IBS2)), file=f)
        
        
        print('AEAdv Model Ctd Avg, '+str(np.mean(CtD3)), file=f)
        print('AEAdv Model Ctd StdDev, '+str(np.std(CtD3)), file=f)
        print('AEAdv Model IBS Avg, '+str(np.mean(IBS3)), file=f)
        print('AEAdv Model IBS StdDev, '+str(np.std(IBS3)), file=f)
        
        print('AENoAdv Model Ctd Avg, '+str(np.mean(CtD4)), file=f)
        print('AENoAdv Model Ctd StdDev, '+str(np.std(CtD4)), file=f)
        print('AENoAdv Model IBS Avg, '+str(np.mean(IBS4)), file=f)
        print('AENoAdv Model IBS StdDev, '+str(np.std(IBS4)), file=f)
        
        print('MSTDImpute Model Ctd Avg, '+str(np.mean(CtD5)), file=f)
        print('MSTDImpute Model Ctd StdDev, '+str(np.std(CtD5)), file=f)
        print('MSTDImpute Model IBS Avg, '+str(np.mean(IBS5)), file=f)
        print('MSTDImpute Model IBS StdDev, '+str(np.std(IBS5)), file=f)
