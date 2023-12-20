#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:18:42 2023

@author: sara
"""
import os,sys
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))
import torch
import torch.utils.data

from multisurvOrig.src import utils
from PhasedAAE.PhasedAAEModels import FC,DNADecoder,MRNAEncoder,MRNADecoder
import pickle
import pandas as pd
import argparse
import shutil


def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
   
    options.add_argument('-dataLocation', '--dataLocation', action="store", dest="dataLocation",default=None)
    options.add_argument('-modelLocation', '--modelLocation', action="store", dest="modelLocation",default=None)
    options.add_argument('-modelLocEpoch', '--modelLocEpoch', action="store", dest="modelLocEpoch", default=-1, type=int)
    options.add_argument('-batchSize', '--batchSize', action="store", dest="batchSize", default=64, type=int)
    
   
    return options.parse_args()

def DoImputation(modelLocation,modelLocEpoch,dataLocation,batchSize):

    netEncA = FC(5000, 512, 5, scaling_factor=2)
    netDecA = DNADecoder(inFeatures=512)
    netEncB = MRNAEncoder(in_features=1000,latentdim=512,simple=True)
    netDecB = MRNADecoder(inFeatures=512,outFeatures=1000)
    
    netEncA.load_state_dict(torch.load(os.path.join(modelLocation,"netEncA_%s.pth" % modelLocEpoch)))
    netDecA.load_state_dict(torch.load(os.path.join(modelLocation,"netDecA_%s.pth" % modelLocEpoch)))
    netEncB.load_state_dict(torch.load(os.path.join(modelLocation,"netEncB_%s.pth" % modelLocEpoch)))
    netDecB.load_state_dict(torch.load(os.path.join(modelLocation,"netDecB_%s.pth" % modelLocEpoch)))
    
    netEncA.eval()
    netDecA.eval()
    netEncB.eval()
    netDecB.eval()
    
    
    datasets=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),modalities=["clinical","mRNA",'DNAm'], batch_size=batchSize,return_patient_id=True)
    
    os.makedirs(dataLocation+"Imputed", exist_ok=True)
    dataLocationNew=dataLocation+"Imputed"
    shutil.copytree(os.path.join(dataLocation,'RNA-seq'),os.path.join(dataLocationNew,'RNA-seq'))
    shutil.copytree(os.path.join(dataLocation,'DNAm'),os.path.join(dataLocationNew,'DNAm'))
    shutil.copytree(os.path.join(dataLocation,'Clinical'),os.path.join(dataLocationNew,'Clinical'))
    shutil.copyfile(os.path.join(dataLocation,'labels.tsv'),os.path.join(dataLocationNew,'labels.tsv'))
    dataLocation=dataLocationNew
    
    for datasetType in ["train","val","test"]:
        for patid in range(len(datasets[datasetType].patient_ids)):
            (dataPat,*rest)=datasets[datasetType].__getitem__(patid)
            datamRNA=dataPat["mRNA"]
            dataDNAm=dataPat["DNAm"]
            print(patid)
            if torch.sum(datamRNA)==0.0 and  torch.sum(dataDNAm)!=0.0:
                with torch.no_grad():
                    dataDNAm=torch.unsqueeze(dataDNAm,0)
                    latentA=netEncA(dataDNAm)
                    imputedB=netDecB(latentA)
                    imputedB=torch.squeeze(imputedB,0)
                dataPat["mRNA"]=imputedB
                patientID=rest[-1]
                imputedB_np = imputedB.numpy() #convert to Numpy array
                df = pd.DataFrame(imputedB_np) #convert to a dataframe
                df.to_csv(os.path.join(dataLocation,'RNA-seq',patientID+'.tsv'), sep="\t",index=False,header=False) #save to file
                
            if torch.sum(dataDNAm)==0.0 and  torch.sum(datamRNA)!=0.0:
                with torch.no_grad():
                    datamRNA=torch.unsqueeze(datamRNA,0)
                    latentB=netEncB(datamRNA)
                    imputedA=netDecA(latentB)
                    imputedA=torch.squeeze(imputedA,0)
                dataPat["DNAm"]=imputedA
                patientID=rest[-1]
                imputedA_np = imputedA.numpy() #convert to Numpy array
                df = pd.DataFrame(imputedA_np) #convert to a dataframe
                df.to_csv(os.path.join(dataLocation,'DNAm','5k',patientID+'.tsv'), sep="\t",index=False,header=False) #save to file
         
            
            


if __name__ == '__main__':
    if os.getcwd() not in sys.path:
        sys.path.append(os.path.join(os.getcwd()))
    args = setup_args()
    DoImputation(args.modelLocation,args.modelLocEpoch,args.dataLocation,args.batchSize)
