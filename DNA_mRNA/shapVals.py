#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:51:51 2023

@author: sara
"""
import sys
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))
import pickle

# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
import torch
from PhasedAAE.PhasedAAEModels import ClinicalEncoder2,Risk_Predictor,MRNAEncoder,ClinicalDecoder2,MRNADecoder,Adv_Classifier,FC,DNADecoder


# from multisurvOrig.src import dataset
# from multisurvOrig.src.model import Model
# from multisurvOrig.src import utils
# from multisurvOrig.src import evaluation
import argparse
from time import time as gettime
import shap

import torch.nn as nn
import torch

class FullModel(nn.Module):
    def __init__(self, netMSEnc,netMSClf):
        super().__init__()
        self.msEnc=netMSEnc
        self.msClf=netMSClf

    def forward(self, x):
        x=self.msEnc(x)
        risk=self.msClf(x)
        return risk

def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('-weightDir', '--weightDir', action="store", dest="weightDir",default=None)
    options.add_argument('-datasetDir', '--datasetDir', action="store", dest="datasetDir",default=None)
    
    return options.parse_args()
   
# device='cpu'
# modalities=['mRNA']
# exclude_patients=[]


# dataLocationTrain='/Users/sara/Desktop/multisurvAll/data/dataForModelsmp60'
# labels = pd.read_csv(os.path.join(dataLocationTrain,'labels.tsv'), sep='\t')
# datasets=utils.get_dataloaders(data_location=dataLocationTrain, labels_file=os.path.join(dataLocationTrain,'labels.tsv'),modalities=modalities, batch_size=16,exclude_patients=exclude_patients,return_patient_id=True)
# dataloaderTrain = torch.utils.data.DataLoader(datasets['train'], batch_size=1000,shuffle=True, drop_last=False)
# datasetTrain=next(iter(dataloaderTrain))
# fileName="datasetTrain1000Shap"
# with open(os.path.join('/Users/sara/Desktop/mp60models/',fileName),'wb') as file:
#     pickle.dump(datasetTrain,file)


# dataLocationTest='/Users/sara/Desktop/multisurvAll/data/dataForModels'
# labels = pd.read_csv(os.path.join(dataLocationTest,'labels.tsv'), sep='\t')
# datasets=utils.get_dataloaders(data_location=dataLocationTest, labels_file=os.path.join(dataLocationTest,'labels.tsv'),modalities=modalities, batch_size=16,exclude_patients=exclude_patients,return_patient_id=True)
# dataloaderTest = torch.utils.data.DataLoader(datasets['test'], batch_size=len(datasets['test']), shuffle=True, drop_last=False)
# datasetTest=next(iter(dataloaderTest))
# fileName="dataTestFullShap"
# with open(os.path.join('/Users/sara/Desktop/mp60models/',fileName),'wb') as file:
#     pickle.dump(datasetTest,file)
#keep above datasets consistent
################################################################



#INPUTS:
#'/Users/sara/Desktop/mp60models/bestRNAImpute'

###########

def getShapValues(datasetDir,weightDir,fileNameTrain="datasetTrain1000Shap",fileNameTest="dataTestFullShap",shapOutputLabel="All"):
    #startTime=time.time()
    print(shap.__version__)
    startTime= gettime()
    
    with open(os.path.join(datasetDir,fileNameTrain),'rb') as file:
        datasetTrain=pickle.load(file)
    
    
    with open(os.path.join(datasetDir,fileNameTest),'rb') as file:
        datasetTest=pickle.load(file)

    (data,time,event,*rest)=datasetTrain
    dataTrain=data["mRNA"]
    print(len(dataTrain))
    
    (data,time,event,*rest)=datasetTest
    dataTest=data["mRNA"]
    print(len(dataTest))

    netMSEnc=MRNAEncoder(in_features=1000,latentdim=512,simple=True,VAE=False)
    netMSClf = Risk_Predictor(nz=512,n_out=30)
    netMSEnc.load_state_dict(torch.load(os.path.join(weightDir,"netMSEnc.pth")))
    netMSClf.load_state_dict(torch.load(os.path.join(weightDir,"netMSClf.pth")))
    fullModel=FullModel(netMSEnc, netMSClf)
    fullModel.eval() #SHOULD THIS BE IN EVAL MODE
    e = shap.DeepExplainer(fullModel,dataTrain)
    shap_values=e.shap_values(dataTest)


    with open(os.path.join(weightDir,"shapValues"+shapOutputLabel),'wb') as file:
        pickle.dump(shap_values,file)
    
    endTime=gettime()
    print(str(endTime-startTime))
#cancer_types=['THCA']
#exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])




#shap.summary_plot(shap_values[0],dataTest,feature_names=genes, plot_type="bar")

if __name__ == '__main__':
    args = setup_args()
    
    getShapValues(args.datasetDir,args.weightDir)

