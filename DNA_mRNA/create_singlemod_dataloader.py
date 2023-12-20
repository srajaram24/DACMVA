#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:55:08 2023

@author: sara
"""

import os,sys
os.chdir('/Users/sara/Desktop/modalimputation')
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))
from multisurv.src import utilsMod2

import pickle
import random

import shutil
import pandas as pd

for b in [4]:
    batchSize=b
    missingPercent=0.0
    cancerTypeLabel="All"
    cancerTypeSubset=["BLCA","BRCA","HNSC","KIRC","LGG","LIHC","LUAD","LUSC","OV","STAD"]#['ACC','KICH','KIRC','KIRP','LGG','LIHC','LUAD','MESO','PAAD','PRAD','SKCM','UCEC','THCA']
    modality="mRNA"
    dataLoaderLoc='/Users/sara/Desktop/dataloadersmRNADNA/coxmp0dl'
    
    
    if missingPercent==0.0:
        dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModels'
    else:
        dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModelsmp'+str(int(missingPercent*100))
        
    if cancerTypeSubset[0]=="All":
        exclude_patients = []
    else:
        cancer_types=cancerTypeSubset
        labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
        exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])
    
    dataloaderTrain,dataloaderVal,dataloaderTest=utilsMod2.get_dataloadersSingleMod(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'), modality=modality,exclude_patients=exclude_patients,return_patient_id=True,batch_size=batchSize)
    
    
    if modality=="mRNA":
        mod="RNA"
    elif modality=="DNAm":
        mod="DNA"
    else:
        ValueError("modality not defined")
        
    fileName="dataloader"+cancerTypeLabel+mod+"Train"+str(batchSize)+"mp"+str(int(missingPercent*100))
    with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
        pickle.dump(dataloaderTrain,file)
        
    fileName="dataloader"+cancerTypeLabel+mod+"Val"+str(batchSize)+"mp"+str(int(missingPercent*100))
    with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
        pickle.dump(dataloaderVal,file)
        
    fileName="dataloader"+cancerTypeLabel+mod+"Test"+str(batchSize)+"mp"+str(int(missingPercent*100))
    with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
        pickle.dump(dataloaderTest,file)
