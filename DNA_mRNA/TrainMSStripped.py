#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:36:39 2023

@author: sara
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:46:08 2022

@author: sara
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
sys.path.append(os.getcwd())
#print(sys.path)
import shutil

from torch.utils.tensorboard import SummaryWriter


from PhasedAAE.PhasedAAEModels import MRNAEncoder as MRNAEncoderForMS
from PhasedAAE.PhasedAAEModels import Risk_Predictor as Risk_PredictorForMS
from multisurv.src.loss import Loss as MultisurvLoss
from pycox.evaluation import EvalSurv
from AAEmiRNA.TestReconLoss2 import getTestReconLoss
from TrainUnimodalMS.TestUnimodal import getTestResultsUnimodal
import pickle
from ray import tune
from trainingHelperFunctions import accuracy,_predictions_to_pycox
import math
from DNA_mRNA.EvaluateMSAAE import EvaluateMSforAAE
import random

torch.manual_seed(42)


#============ PARSE ARGUMENTS =============






class TrainMSStrippedClass:
    def __init__(self,config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,trainLoader=None,valLoader=None):
        
        
        self.trainLoader=trainLoader
        self.valLoader=valLoader
        folderName="bs_{}_lrMS_{}_{}".format(config["batch_size"],config["lrMS"],config["cancLabel"]) #here
        save_dir=os.path.join(save_dir,folderName)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir,"modelfiles"), exist_ok=True)
        os.makedirs(os.path.join(save_dir,"tensorboardlogs"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir,"tensorboardlogs"))
        
        

        self.netMSEnc=MRNAEncoderForMS(in_features=1000,latentdim=512,simple=True,VAE=False)
        self.netMSClf = Risk_PredictorForMS(nz=512,n_out=30)
        
       
        
        self.output_intervals=torch.arange(0., 365 * 31, 365).double() #NEEDED for computing the multisurv loss
        
       
        
        if gpu>-1:
            self.device = torch.device('cuda:'+str(gpu))
        else:
            self.device=torch.device("cpu")
            
        if gpu>-1:
            self.netMSEnc.to(self.device)
            self.netMSClf.to(self.device)
            self.output_intervals.to(self.device)
       
             
        
        
        # setup optimizer
        self.opt_netMSEnc = optim.Adam(list(self.netMSEnc.parameters()), lr=config["lrMS"])
        self.opt_netMSClf = optim.Adam(list(self.netMSClf.parameters()), lr=config["lrMS"])
    
        
        # loss criteria
        self.criterion_MultiSurv = MultisurvLoss()

        
        self.config=config
        self.missingPercent=missingPercent
        self.dataloaderDir=dataloaderDir
        self.gpu=gpu
        self.max_epochs=max_epochs
        self.save_dir=save_dir
        self.dataLocationTest=dataLocationTest
        # setup logger
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            print(config, file=f)
            print(self.netMSEnc, file=f)
            print(self.netMSClf, file=f)
            print("missing percent is "+str(self.missingPercent))
                

    
    def trainMS(self,origBatch):
        self.netMSEnc.train()
        self.netMSClf.train()
        # reset parameter gradients
        self.netMSEnc.zero_grad() 
        self.netMSClf.zero_grad() 
       
        (data,timeOrig,eventOrig,*rest)=origBatch
        dataOrig=data["mRNA"]
        dataOrig = Variable(dataOrig)
        
        numOrig=timeOrig.shape[0]
        if self.gpu>-1:
            dataOrig = dataOrig.to(self.device)

 
        
        # forward pass
        latentsOrig=self.netMSEnc(dataOrig)
        riskOrig = self.netMSClf(latentsOrig)
        msLossOrig = self.criterion_MultiSurv(risk=riskOrig, times=timeOrig, events=eventOrig, breaks=self.output_intervals,device=self.device)
        

        loss = msLossOrig
            
        
    
        # backpropagate and update model
        loss.backward()
        self.opt_netMSEnc.step()
        self.opt_netMSClf.step()
        
        summary_stats = {'msLossOrig': msLossOrig.item()*numOrig}
        summary_stats['numOrig']=numOrig
        return summary_stats
       

    
    

    def trainCM(self):
        #additional params
        
        # load data
        

        if self.trainLoader is None:
                      
            with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"RNAVal"+str(self.config["RNAValSize"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                 dataloaderRNAValCanc = pickle.load(file)
            
            with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"RNATrain"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                 dataloaderRNATrainCanc = pickle.load(file)
        
        else:
            dataloaderRNATrainCanc=self.trainLoader
            dataloaderRNAValCanc=self.valLoader
        ### main training loop
        
        bestCtdMS=0.0
        bestEpoch=-1
        
       
        
        for epoch in range(self.max_epochs):
            print(epoch)
        

            msLossTrain=0.0
            numSamples=0.0

            for idx, batch in enumerate(dataloaderRNATrainCanc):
                
                    
                
                out=self.trainMS(batch)
                msLossTrain += out['msLossOrig']
                numSamples+=out["numOrig"]
               
                
            
    
            
           
            _, _, ibsValMS, _, _, _, lossValMS,concordValMS=getTestResultsUnimodal(self.netMSEnc,self.netMSClf,dataloaderRNAValCanc,self.save_dir,modality="mRNA",gpu=self.gpu,makeTSNEIdx=-1,perplexity=2)
           
            self.writer.add_scalar("concordValMS", concordValMS, epoch)
           
            
            
            self.writer.add_scalar("ibsValMS", ibsValMS, epoch)
            self.writer.add_scalar("lossValMS", lossValMS, epoch)
            msLossTrain /= numSamples
            rRatio=1-(msLossTrain/lossValMS)
            self.writer.add_scalar("rRatio", rRatio, epoch)
            
            # save model
            if concordValMS>bestCtdMS:
                weightDir=os.path.join(self.save_dir,"modelfiles")
                bestCtdMS=concordValMS
                correspondingIBS=ibsValMS
                bestEpoch=epoch
                if os.path.exists(weightDir):
                    shutil.rmtree(weightDir)
                os.makedirs(weightDir, exist_ok=True)
                torch.save(self.netMSEnc.cpu().state_dict(), os.path.join(weightDir,"netMSEnc.pth"))
                torch.save(self.netMSClf.cpu().state_dict(), os.path.join(weightDir,"netMSClf.pth"))
                
                
            if self.gpu>-1:
                self.netMSEnc.to(self.device)
                self.netMSClf.to(self.device)
  
       
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            print("Best epoch for MS Val "+str(bestEpoch), file=f)

        
        ctdTestResults,ibsTestResults=EvaluateMSforAAE(self.dataLocationTest,weightDir,self.gpu,self.config["cancer_type"],evaluateMulti=False)
        self.writer.add_scalar("TestingCtD"+self.config["cancLabel"], ctdTestResults, bestEpoch)
        self.writer.add_scalar("TestingIBS"+self.config["cancLabel"], ibsTestResults, bestEpoch)
        self.writer.flush()
        self.writer.close()
        return bestCtdMS,correspondingIBS

