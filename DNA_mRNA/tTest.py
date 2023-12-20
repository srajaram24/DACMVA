#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:05:19 2023

@author: sara
"""
import torch
import torch.utils.data

from torch.autograd import Variable

import os,sys
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))
import argparse

from multisurv.src.loss import Loss as MultisurvLoss
import pandas as pd
from pycox.evaluation import EvalSurv

from PhasedAAE.PhasedAAEModels import ClinicalEncoder2,Risk_Predictor,MRNAEncoder,ClinicalDecoder2,MRNADecoder,Adv_Classifier,FC,DNADecoder
import sys
import pickle
from multisurvOrig.src import utils
import scipy
from sklearn.utils import resample

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options

    options.add_argument('-dataLocationTest', '--dataLocationTest', action="store", dest="dataLocationTest",default=None)
    options.add_argument('-weightDirBase', '--weightDirBase', action="store", dest="weightDirBase",default=None)
    options.add_argument('-weightDirBest', '--weightDirBest', action="store", dest="weightDirBest",default=None)
    options.add_argument('-cancerTypeSubsetPath', '--cancerTypeSubsetPath', action="store", dest="cancerTypeSubsetPath",default=None)
    options.add_argument('-tag', '--tag', action="store", dest="tag",default=None)
    

    return options.parse_args()


     
     


def _predictions_to_pycox(preds, time_points=None):
    # Convert to survival probabilities
    surv_probs = torch.cumprod(preds, 1)
    df = pd.DataFrame(torch.transpose(surv_probs, 0, 1).cpu().numpy())

    if time_points is None:
        time_points = torch.arange(0.5, 30, 1)

    # Replace automatic index by time points
    df.insert(0, 'time', time_points)
    df = df.set_index('time')

    return df

def getRisks(data,time,event,netEnc,netCondClf,criterion_MultiSurv,thisDevice,modality):
    if modality=="clinical":
        dataACat=data["clinical"][0]
        dataACont=data["clinical"][1]
        dataACat, dataACont = Variable(dataACat), Variable(dataACont)
        dataA=(dataACat,dataACont)
        data=dataA
    else:
        dataB=data[modality]
        dataB = Variable(dataB)
        data=dataB
    
    output_intervals=torch.arange(0., 365 * 31, 365).double()
    
    if thisDevice != "cpu":
        data=data.to(thisDevice)
        output_intervals=output_intervals.to(thisDevice)
    # forward pass
    with torch.no_grad():
        allEncOutputs=netEnc(data)
        if type(allEncOutputs) is tuple: 
            latents=allEncOutputs[0]
        else:
            latents=allEncOutputs
        risk = netCondClf(latents)
        # compute losses
        clf_class_loss = criterion_MultiSurv(risk=risk, times=time, events=event, breaks=output_intervals,device=thisDevice)
    
    return risk,clf_class_loss,latents






def getWilcoxResults(dataLocationTest,weightDirBase,weightDirBest,cancerTypeSubsetPath,tag):
    
    modality="mRNA"
    
    with open(os.path.join(cancerTypeSubsetPath,"cancerTypeSubset"), 'rb') as file:
         cancerTypeSubset = pickle.load(file)
    if cancerTypeSubset[0]=="All":
        exclude_patients = []
    else:
        cancer_types=cancerTypeSubset
        labels = pd.read_csv(os.path.join(dataLocationTest,'labels.tsv'), sep='\t')
        exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])
    
    with open(os.path.join(weightDirBest,'wilcoxTest'+tag+'.txt'),'w') as f:
        print(weightDirBest+" Compared with "+weightDirBase,file=f)
        
        
    netMSEncBase=MRNAEncoder(in_features=1000,latentdim=512,simple=True,VAE=False)
    netMSClfBase = Risk_Predictor(nz=512,n_out=30)
    netMSEncBase.load_state_dict(torch.load(os.path.join(weightDirBase,"netMSEnc.pth")))
    netMSClfBase.load_state_dict(torch.load(os.path.join(weightDirBase,"netMSClf.pth")))

    netMSEncBest=MRNAEncoder(in_features=1000,latentdim=512,simple=True,VAE=False)
    netMSClfBest = Risk_Predictor(nz=512,n_out=30)
    netMSEncBest.load_state_dict(torch.load(os.path.join(weightDirBest,"netMSEnc.pth")))
    netMSClfBest.load_state_dict(torch.load(os.path.join(weightDirBest,"netMSClf.pth")))

    
    # if batchSize<0:
    #     numRepeats=1
    #     bs=32
    # else:
    #     numRepeats=5
    #     bs=batchSize
    
    numRepeats=1000
    datasets=utils.get_dataloaders(data_location=dataLocationTest, labels_file=os.path.join(dataLocationTest,'labels.tsv'),modalities=[modality], batch_size=32,exclude_patients=exclude_patients,return_patient_id=True)
    RNATest = torch.utils.data.DataLoader(datasets['test'], batch_size=32,shuffle=True, drop_last=False)
    
    
    # if batchSize<0:
    #     RNALoader=[]
    #     for i in range(1000):
    #         RNALoader.append(resample(RNATest.dataset))
    #     print("completed RNA loader")
            
    # else:
    #     RNALoader=RNATest
    
    with open(os.path.join(weightDirBest,'wilcoxTest'+tag+'.txt'),'a') as f:
        print('Len of RNA test set is '+str(len(RNATest.dataset)),file=f)
        
    
    
    
    criterion_MultiSurv = MultisurvLoss()
    
    
    netMSEncBase.eval()
    netMSClfBase.eval()
    netMSEncBest.eval()
    netMSClfBest.eval()
    
    
    resultsBaseC=[]
    resultsBestC=[]
    for idx in range(numRepeats):
        print(idx)
        batch=resample(RNATest.dataset)
        batchDL = torch.utils.data.DataLoader(batch, batch_size=len(batch),shuffle=False, drop_last=False)
        batch=next(iter(batchDL))
        
        running_durationsBase = torch.FloatTensor()
        running_censorsBase = torch.LongTensor()
        running_risksBase = torch.FloatTensor()
    
        running_durationsBest = torch.FloatTensor()
        running_censorsBest = torch.LongTensor()
        running_risksBest = torch.FloatTensor()
        (data,*rest)=batch
        
        time=rest[0]
        event=rest[1]
        
        risk,_,_=getRisks(data,time,event,netMSEncBase,netMSClfBase,criterion_MultiSurv,'cpu',modality=modality)
    
        running_durationsBase = torch.cat((running_durationsBase,time.data.float()))
        running_censorsBase = torch.cat((running_censorsBase,event.long().data))
        running_risksBase = torch.cat((running_risksBase, risk.cpu()))
        
        
        risk,_,_=getRisks(data,time,event,netMSEncBest,netMSClfBest,criterion_MultiSurv,'cpu',modality=modality)
    
        running_durationsBest = torch.cat((running_durationsBest,time.data.float()))
        running_censorsBest = torch.cat((running_censorsBest,event.long().data))
        running_risksBest = torch.cat((running_risksBest, risk.cpu()))
        
        
        surv_probsBase = _predictions_to_pycox(running_risksBase, time_points=None)
        running_durationsBase = running_durationsBase.cpu().numpy()
        running_censorsBase = running_censorsBase.cpu().numpy()
        ev=EvalSurv(surv_probsBase, running_durationsBase, running_censorsBase,censor_surv='km')
        concordBase= ev.concordance_td('adj_antolini')
        
        surv_probsBest = _predictions_to_pycox(running_risksBest, time_points=None)
        running_durationsBest = running_durationsBest.cpu().numpy()
        running_censorsBest = running_censorsBest.cpu().numpy()
        ev=EvalSurv(surv_probsBest, running_durationsBest, running_censorsBest,censor_surv='km')
        concordBest= ev.concordance_td('adj_antolini')
        resultsBaseC.append(concordBase)
        resultsBestC.append(concordBest)
    
    wilcResult=scipy.stats.wilcoxon(resultsBestC,resultsBaseC,alternative='greater')
    with open(os.path.join(weightDirBest,'wilcoxTest'+tag+'.txt'),'a') as f:
        print(wilcResult,file=f)
    print(wilcResult)
            
# for sample in RNATest.dataset:
#     (data,time,event,*rest)=sample
#     running_durations = torch.FloatTensor()
#     running_censors = torch.LongTensor()
#     running_risks = torch.FloatTensor()
    
#     running_durations = torch.cat((running_durations,time.data.float()))
#     running_censors = torch.cat((running_censors,event.long().data))
#     running_risks = torch.cat((running_risks, risk.cpu()))
# time_grid = np.linspace(running_durationsBase.min(), running_durationsBase.max(), 100)
# drop_last_times=25
# time_grid = time_grid[:-drop_last_times]
# ibsPaired = evBase.integrated_brier_score(time_grid)
   
            
    
if __name__ == '__main__':
    args = setup_args()

    getWilcoxResults(args.dataLocationTest, args.weightDirBase, args.weightDirBest,args.cancerTypeSubsetPath,args.tag)
  
 

  
    


    
