#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:32:06 2023

@author: sara
"""
from multisurv.src import utils
import os
import torch
import pandas as pd

dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModelsmp30'
saveDir='/Users/sara/Desktop/modalimputation/'
labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')

allCancers = ['BLCA', 'BRCA', 'CESC', 'COAD', 'READ',
            'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML',
            'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
            'PAAD', 'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']

with open(os.path.join(saveDir,'DNARNAImbalances.txt'),'w') as f:
    line="Cancer"+","+"dataType"+","+"numPairs"+","+"numMRNA"+","+"numDNA"+","+"frac"
    print(line,file=f)

for aCancer in allCancers:
    print("=====================================")
    print(aCancer)
    cancers=[aCancer]
    exclude_cancers = list(labels.loc[~labels['project_id'].isin(cancers), 'submitter_id'])
    dataloaders,datasets=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),modalities=["mRNA",'DNAm'], batch_size=32, exclude_patients=exclude_cancers)

    

    for dataType in ["train","val","test"]:
        numMRNA=0
        numDNA=0
        numPairs=0
        dataset=datasets[dataType]
        for patid in range(len(dataset.patient_ids)):
            (dataPat,*rest)=dataset.__getitem__(patid)
            #dataClin=dataPat["clinical"][0]
            datamRNA=dataPat["mRNA"]
            dataDNA=dataPat["DNAm"]
            
            if torch.sum(datamRNA)!=0.0:
                numMRNA+=1
            if torch.sum(dataDNA)!=0.0:
                numDNA+=1
            if torch.sum(dataDNA)!=0.0 and torch.sum(datamRNA)!=0.0:
                numPairs+=1
        
        with open(os.path.join(saveDir,'DNARNAImbalances.txt'),'a') as f:
            line=aCancer+","+dataType+","+str(numPairs)+","+str(numMRNA)+","+str(numDNA)+","+str(numMRNA/numDNA)
            print(line,file=f)
        print(aCancer)
        print(dataType)
        print("numMRNA "+str(numMRNA))
        print("numDNA "+str(numDNA))
        print("=====================================")



