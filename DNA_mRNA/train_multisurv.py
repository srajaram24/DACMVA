#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:58:42 2023

@author: sara
"""

import sys
import os
# Make modules in "src" dir visible

if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))

import torch
from multisurvOrig.src import utils
from multisurvOrig.src.model import Model
from multisurvOrig.src import evaluation

import argparse
import itertools

import pandas as pd

from DNA_mRNA.EvaluateByCancerSingleMod import evaluateByCancerTypeSingleMod


    
def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-save_dir', '--save_dir', action="store", dest="save_dir",default=None)
    options.add_argument('-fusion', '--fusion', action="store", dest="fusion",default='max')
    options.add_argument('-dataLocation', '--dataLocation', action="store", dest="dataLocation",default=None)
    options.add_argument('-modelLocation', '--modelLocation', action="store", dest="modelLocation",default=None)
    options.add_argument('-modelLocEpoch', '--modelLocEpoch', action="store", dest="modelLocEpoch",default=-1,type=int)
    options.add_argument('-numEpochs', '--numEpochs', action="store", dest="numEpochs", default=40, type=int)
    options.add_argument('-learningRate', '--learningRate', action="store", dest="learningRate", default=0.0001, type=float)
    options.add_argument('-batchSize', '--batchSize', action="store", dest="batchSize", default=32, type=int)
    options.add_argument('-lungOnly', '--lungOnly', action="store_true", dest="lungOnly",default=False)
    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
    options.add_argument('-dataLoaderLocPaired', '--dataLoaderLocPaired', action="store", dest="dataLoaderLocPaired",default=None)
    options.add_argument('-dataLoaderLocUnpairedA', '--dataLoaderLocUnpairedA', action="store", dest="dataLoaderLocUnpairedA",default=None)
    options.add_argument('-dataLoaderLocUnpairedB', '--dataLoaderLocUnpairedB', action="store", dest="dataLoaderLocUnpairedB",default=None)
    options.add_argument('-dataLocationTest', '--dataLocationTest', action="store", dest="dataLocationTest",default=None)
    return options.parse_args()

def trainMS(save_dir,dataLocation,learningRate,batchSize,numEpochs,fusion,modelLocation,modelLocEpoch,
         dataLoaderLocPaired,dataLoaderLocUnpairedA,dataLoaderLocUnpairedB,lungOnly,gpu,dataLocationTest):
    #initialize summary writer for tensorboard
    
    
    if lungOnly:
        cancer_types= ['LUAD', 'LUSC']
        labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
        exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])
    else:
        exclude_patients = []
        # cancer_types = ['BRCA', 'BLCA', 'HNSC', 'KIRC',
        #             'LGG', 'LIHC', 'LUAD', 'LUSC', 
        #              'OV', 'STAD']
        # labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
        # exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])
    
    lrFactor=0.5
    
   
    modalities=['mRNA']
    

    
    os.makedirs(save_dir, exist_ok=True)
    save_dir=os.path.join(save_dir,'lr_{}_bs_{}_{}'.format(learningRate,batchSize,fusion))
    os.makedirs(save_dir, exist_ok=True)
    if modelLocation is None:
        modelLocationForLog="No Pretrained Model Loaded"
    else:
        modelLocationForLog=modelLocation
    with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
        print("modelLocation is "+str(modelLocationForLog)+","+"modelLocEpoch is "+str(modelLocEpoch), file=f)
        
    os.makedirs(os.path.join(save_dir,'training_logs/'), exist_ok=True)
    
    datasets=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),modalities=modalities, batch_size=batchSize,exclude_patients=exclude_patients,return_patient_id=True)
    dataloaders = {'train': torch.utils.data.DataLoader(
        datasets['train'], batch_size=batchSize,
        shuffle=True, num_workers=4, drop_last=True),
                    'val': torch.utils.data.DataLoader(
        datasets['val'], batch_size=batchSize,
        shuffle=True, num_workers=4, drop_last=True),
                    'test': torch.utils.data.DataLoader(
        datasets['test'], batch_size=16,
        shuffle=True, num_workers=4, drop_last=True)}
    
  
    if gpu>-1:
        device = torch.device('cuda:'+str(gpu))
    else:
        device=torch.device("cpu")
    
 
    
    multisurv = Model(dataloaders=dataloaders,device=device,fusion_method=fusion,fcBlock=False)
    print(multisurv.model)
    if modelLocation is not None:
        if modelLocEpoch>-1:
            mRNAPretrain=torch.load(os.path.join(modelLocation,"netEncB_%s.pth" % modelLocEpoch))
            mRNAPretrain = {k: v for k, v in mRNAPretrain.items() if k in multisurv.model.mRNA_submodel.state_dict().keys()}
            multisurv.model.mRNA_submodel.load_state_dict(mRNAPretrain)
            DNAPretrain=torch.load(os.path.join(modelLocation,"netEncA_%s.pth" % modelLocEpoch))
            DNAPretrain = {k: v for k, v in DNAPretrain.items() if k in multisurv.model.DNAm_submodel.state_dict().keys()}
            multisurv.model.DNAm_submodel.load_state_dict(DNAPretrain)
        else:
            weightDir=os.path.join(modelLocation,"multisurv.pth")
            multisurv.model.load_state_dict(torch.load(weightDir))
        print("LOADED PRETRAINED MODELS")
    
        for child in multisurv.model.mRNA_submodel.children():
           for param in child.parameters():
               param.requires_grad = False
        for child in multisurv.model.DNAm_submodel.children():
           for param in child.parameters():
               param.requires_grad = False
    
    fit_args = {'lr': learningRate,'num_epochs': numEpochs,'info_freq': 1,'lr_factor': lrFactor,'scheduler_patience': 10,
                'log_dir': os.path.join(save_dir,'training_logs/'),}
    
    multisurv.fit(**fit_args)
    
    print("=====================================")
    with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
        print("multisurv.best_model_weights.keys()",file=f)
        print(multisurv.best_model_weights.keys(),file=f)
        print("multisurv.best_concord_values",file=f)
        print(multisurv.best_concord_values,file=f)
    torch.save(multisurv.best_model_weights[list(multisurv.best_model_weights.keys())[0]], os.path.join(save_dir,"multisurv.pth"))
    print("####################################")
    
    bestEpoch=int(list(multisurv.best_model_weights.keys())[0].replace('epoch',''))
    
    evaluateByCancerTypeSingleMod(dataLocationTest,save_dir,gpu=gpu,bestEpoch=bestEpoch)
        
if __name__ == '__main__':
    if os.getcwd() not in sys.path:
        sys.path.append(os.path.join(os.getcwd()))
    args = setup_args()
    
    if args.batchSize==-1:
        for bs in [32,64]:
            for lr in [2e-5,5e-5,1e-4,1e-5]:
                if lr==1e-5 and bs==32:
                    continue
                trainMS(args.save_dir,args.dataLocation,lr,bs,args.numEpochs,args.fusion,modelLocation=args.modelLocation,modelLocEpoch=args.modelLocEpoch,
                         dataLoaderLocPaired=args.dataLoaderLocPaired,dataLoaderLocUnpairedA=args.dataLoaderLocUnpairedA,dataLoaderLocUnpairedB=args.dataLoaderLocUnpairedB,lungOnly=args.lungOnly,gpu=args.gpu,dataLocationTest=args.dataLocationTest)

    else:
        trainMS(args.save_dir,args.dataLocation,args.learningRate,args.batchSize,args.numEpochs,args.fusion,modelLocation=args.modelLocation,modelLocEpoch=args.modelLocEpoch,
                 dataLoaderLocPaired=args.dataLoaderLocPaired,dataLoaderLocUnpairedA=args.dataLoaderLocUnpairedA,dataLoaderLocUnpairedB=args.dataLoaderLocUnpairedB,lungOnly=args.lungOnly,gpu=args.gpu,dataLocationTest=args.dataLocationTest)


