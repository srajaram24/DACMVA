#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:52:22 2022

@author: sara
"""


import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable


from warnings import simplefilter


import sys


simplefilter(action='ignore', category=FutureWarning)


def getRecon(thisBatch,netEnc,netDec,criterion_reconstruct,modality,use_gpu=False):
    (data,time,event,*rest)=thisBatch
    
    dataInput=data[modality]
    dataInput = Variable(dataInput)
 
    
    if use_gpu:
        dataInput=dataInput.cuda()
    # forward pass
    with torch.no_grad():
        latents,_,_=netEnc(dataInput)
        recon=netDec(latents)
        recon_loss = criterion_reconstruct(dataInput, recon)
        
    return recon_loss




def getTestReconLoss(netEnc,netDec,dataloaderPairedTest,save_dir,modality,use_gpu=False,dataloaderUnpairedTest=None):

    criterion_reconstruct = nn.MSELoss()
    
    netEnc.eval()
    netDec.eval()
    
    reconLossPaired = 0
    reconLossUnpaired = 0

    nPaired = 0
    nUnpaired = 0
    
    
    for idx, batch in enumerate(dataloaderPairedTest):
        (data,*rest)=batch
        time=rest[0]
        
        reconLoss=getRecon(batch,netEnc,netDec,criterion_reconstruct,modality,use_gpu)
        reconLossPaired += reconLoss*time.shape[0]
        nPaired += time.shape[0]
  
    
    if dataloaderUnpairedTest is not None:
        for idx, batch in enumerate(dataloaderUnpairedTest):
            (data,*rest)=batch
            time=rest[0]
            
            reconLoss=getRecon(batch,netEnc,netDec,criterion_reconstruct,modality,use_gpu)
            reconLossUnpaired += reconLoss*time.shape[0]
            nUnpaired += time.shape[0]
        
    
        reconLossTotal=(reconLossPaired+reconLossUnpaired)/(nPaired+nUnpaired)
        reconLossPaired /= nPaired
        reconLossUnpaired /= nUnpaired
    
    else:
        reconLossUnpaired=-1.0
        reconLossPaired /= nPaired
        reconLossTotal=reconLossPaired
    
    
    return reconLossPaired, reconLossUnpaired, reconLossTotal
    


    
