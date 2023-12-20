
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
from TrainAAEIndiv.DNAmRNAModelsExpt import MRNADecoder,DNAEncoder
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






class TrainmRNAImputeClass:
    def __init__(self,config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,modelLocation,fixedConfig=False,trainLoader=None,valLoader=None):
        
        self.fixedConfig=fixedConfig
        self.trainLoader=trainLoader
        self.valLoader=valLoader
        
        UnpLabel=config["UnpairedDNA"]*1 
        folderName="bs_{}_gam_{}_lrMS_{}_t_{}_{}Imp{}Unp{}{}".format(config["batch_size"],config["gamma"],config["lrMS"],config["t"],config["cancLabel"],config["ImputeBS"],UnpLabel,config["tag"])
        save_dir=os.path.join(save_dir,folderName)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir,"modelfiles"), exist_ok=True)
        os.makedirs(os.path.join(save_dir,"tensorboardlogs"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir,"tensorboardlogs"))
        
        
        self.netEncA = DNAEncoder(latentdim=config["latent_dims"],VAE=config["VAE"]) 
        self.netDecB = MRNADecoder(latentdim=config["latent_dims"]) #MRNADecoder(outFeatures=1000)
        self.netMSEnc=MRNAEncoderForMS(in_features=1000,latentdim=512,simple=True,VAE=False)
        self.netMSClf = Risk_PredictorForMS(nz=512,n_out=30)
        
        if modelLocation is not None:
            self.netEncA.load_state_dict(torch.load(os.path.join(modelLocation,"netEncA.pth")))
            self.netDecB.load_state_dict(torch.load(os.path.join(modelLocation,"netDecB.pth")))
        if config["gamma"]!=0.0 and modelLocation is None:
            ValueError("MUST PROVIDE MODEL WITH NONZERO GAMMA")
            return
        
        self.output_intervals=torch.arange(0., 365 * 31, 365).double() #NEEDED for computing the multisurv loss
        
       
        
        if gpu>-1:
            self.device = torch.device('cuda:'+str(gpu))
        else:
            self.device=torch.device("cpu")
            
        if gpu>-1:
            self.netEncA.to(self.device)
            self.netDecB.to(self.device)
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
            print(self.netEncA, file=f)
            print(self.netDecB, file=f)
            print(self.netMSEnc, file=f)
            print(self.netMSClf, file=f)
            print(modelLocation, file=f)
            print("missing percent is "+str(self.missingPercent))
                

    
    def trainMS(self,origBatch,imputedBatch,epochRatio): #here
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
        
        if len(imputedBatch)>0:
            (dataImp,timeImp,eventImp,*rest)=imputedBatch
            dataImp = Variable(dataImp)
          
            numImp=timeImp.shape[0]
           
            if self.gpu>-1:
                dataImp = dataImp.to(self.device)
        
     
            
            # forward pass
            latentsImp=self.netMSEnc(dataImp)
            riskImp = self.netMSClf(latentsImp)
            msLossImp = self.criterion_MultiSurv(risk=riskImp, times=timeImp, events=eventImp, breaks=self.output_intervals,device=self.device)
            
            if epochRatio<0:
                prob=self.config["gamma"]
            else:
                prob=(self.config["gamma"]/(1-math.exp(-1)))*(1-math.exp(-epochRatio))
            #prob=rRatio/self.config["gamma"] #here
            
            summary_stats = {'msLossOrig': msLossOrig.item()*numOrig}
            summary_stats['msLossImp']=msLossImp.item()*numImp
            summary_stats['numImp']=numImp
            summary_stats['numOrig']=numOrig
            loss = (numOrig*msLossOrig+prob*numImp*msLossImp)/(numOrig+prob*numImp)
            
        else:
            msLossImp=0
            numImp=0
            prob=0
            
            summary_stats = {'msLossOrig': msLossOrig.item()*numOrig}
            summary_stats['msLossImp']=0.0
            summary_stats['numImp']=0
            summary_stats['numOrig']=numOrig
            loss = msLossOrig
            
        
    
        # backpropagate and update model
        loss.backward()
        self.opt_netMSEnc.step()
        self.opt_netMSClf.step()
       

    
        return summary_stats
    
    
 
    
    def createImputations(self,dataloaderDNATrain):

        self.netEncA.eval()
        self.netDecB.eval()

        
             
        dnaTrainDataset=dataloaderDNATrain.dataset
      

        imputedDataset=[]

        
        for sample in dnaTrainDataset: 
            (dataPat,*rest)=sample 
            dataDNAm=dataPat["DNAm"]
            if self.gpu>-1:
                dataDNAm=dataDNAm.to(self.device)
            

            with torch.no_grad():
                dataDNAm=torch.unsqueeze(dataDNAm,0)
                if self.config["VAE"]:
                    latentA,_,_=self.netEncA(dataDNAm)
                else:
                    latentA=self.netEncA(dataDNAm)
                imputedB=self.netDecB(latentA).clone().detach()
                imputedB=torch.squeeze(imputedB,0)
                imputedDataset.append((imputedB,*rest))
        
        if len(imputedDataset)>0:
            dl=torch.utils.data.DataLoader(imputedDataset, batch_size=self.config["ImputeBS"],shuffle=False,drop_last=False)
        else:
            ValueError("No imputations!!!!")
 
        return dl
    
    
    def getBatchBelowThresh(self,dlImputed):
        self.netMSEnc.eval()
        self.netMSClf.eval()
        
        imputedDataset=dlImputed.dataset
        numAttempts=0
        batchBelowT=[]
        while len(batchBelowT)<self.config["ImputeBS"] and numAttempts<len(imputedDataset):
            numAttempts+=1
            singleSample=random.choice(imputedDataset)
            (dataImp,timeImp,eventImp,*rest)=singleSample
            if self.gpu>-1:
                dataImp=dataImp.to(self.device)
            with torch.no_grad():
                dataImp=torch.unsqueeze(dataImp,0)
                latentsImp=self.netMSEnc(dataImp)
                riskImp = self.netMSClf(latentsImp).clone().detach()
                timeTensor=torch.tensor([timeImp])
                eventTensor=torch.tensor([eventImp])
                lossVal=self.criterion_MultiSurv(risk=riskImp, times=timeTensor, events=eventTensor, breaks=self.output_intervals,device=self.device)
                if lossVal<self.config["t"]:
                    dataImp=torch.squeeze(dataImp,0)
                    batchBelowT.append((dataImp,timeImp,eventImp,*rest))
        
        if len(batchBelowT)>0:
            dl=torch.utils.data.DataLoader(batchBelowT, batch_size=len(batchBelowT),shuffle=True)
            batchToReturn=next(iter(dl))
        else:
            batchToReturn=[]
            
        return batchToReturn






    def trainCM(self):
        #additional params
        
        # load data
        

        if self.trainLoader is None:
                      
            with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"RNAVal"+str(self.config["RNAValSize"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                 dataloaderRNAValCanc = pickle.load(file)
            
            with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"RNATrain"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                 dataloaderRNATrainCanc = pickle.load(file)
            
            if self.config["UnpairedDNA"]: #batch size here
                with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"Unpaired"+"DNATrain"+str(32)+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                     dataloaderDNATrainCanc = pickle.load(file)
            else:
                with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"DNATrain"+str(32)+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                     dataloaderDNATrainCanc = pickle.load(file)
        else:
            print("LOAD FROM trainLoader and valLoader")
            dataloaderRNATrainCanc=self.trainLoader
            dataloaderRNAValCanc=self.valLoader
            if self.config["UnpairedDNA"]: #batch size here
                with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"Unpaired"+"DNATrain"+str(32)+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                     dataloaderDNATrainCanc = pickle.load(file)
            else:
                with open(os.path.join(self.dataloaderDir,"dataloader"+self.config["cancLabel"]+"DNATrain"+str(32)+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                     dataloaderDNATrainCanc = pickle.load(file)
        ### main training loop
        
        bestCtdMS=0.0
        bestEpoch=-1
        
        if self.config["gamma"]==0:
            dataloaderImputed=[]
        else:
            dataloaderImputed=self.createImputations(dataloaderDNATrainCanc)
        rRatio=0 #here
        for epoch in range(self.max_epochs):
            print(epoch)
        
            msLossOrig=0
            msLossImp=0
            numImp=0
            numOrig=0
            

            for idx, batch in enumerate(dataloaderRNATrainCanc):
                if len(dataloaderImputed)==0:
                    batchImputed=[]
                else:
                    batchImputed=self.getBatchBelowThresh(dataloaderImputed)
        
                out=self.trainMS(batch,batchImputed,-1) #here
                msLossImp += out['msLossImp']
                msLossOrig += out['msLossOrig']
                numImp+=out["numImp"]
                numOrig+=out["numOrig"]
            

            # for idx, batch in enumerate(dataloaderRNATrainCanc):
            #     batchImputed=[]
                    
            
            #     out=self.trainMS(batch,batchImputed,0.0) #here
                
            #     msLossOrig += out['msLossOrig']
                
            #     numOrig+=out["numOrig"]
                
            # for idx in range(0,self.config["maxExtraBatches"]):
            #     if len(dataloaderImputed)==0:
            #         batchImputed=[]
            #     else:
            #         batchImputed=self.getBatchBelowThresh(dataloaderImputed)
                    
            #     out=self.trainMSImputedOnly(batchImputed,-1) #here
            #     msLossImp += out['msLossImp']
            #     numImp+=out["numImp"]
             
                
            
    
            if numImp>0:
                msLossImp /= numImp
            msLossOrig /= numOrig
                
           
            _, _, ibsValMS, _, _, _, lossValMS,concordValMS=getTestResultsUnimodal(self.netMSEnc,self.netMSClf,dataloaderRNAValCanc,self.save_dir,modality="mRNA",gpu=self.gpu,makeTSNEIdx=-1,perplexity=2)
           
            self.writer.add_scalar("concordValMS", concordValMS, epoch)
           
            
            self.writer.add_scalar("msLossImp", msLossImp, epoch)
            self.writer.add_scalar("msLossOrig", msLossOrig, epoch)
            self.writer.add_scalar("numImp", numImp, epoch)
            self.writer.add_scalar("ibsValMS", ibsValMS, epoch)
            self.writer.add_scalar("lossValMS", lossValMS, epoch)
            rRatio=1-(msLossOrig/lossValMS)
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
                torch.save(self.netEncA.cpu().state_dict(), os.path.join(weightDir,"netEncA.pth"))
                torch.save(self.netDecB.cpu().state_dict(), os.path.join(weightDir,"netDecB.pth"))
                torch.save(self.netMSEnc.cpu().state_dict(), os.path.join(weightDir,"netMSEnc.pth"))
                torch.save(self.netMSClf.cpu().state_dict(), os.path.join(weightDir,"netMSClf.pth"))
                
                
            if self.gpu>-1:
                self.netEncA.to(self.device)
                self.netDecB.to(self.device)
                self.netMSEnc.to(self.device)
                self.netMSClf.to(self.device)
            if not self.fixedConfig:
                tune.report()
       
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            print("Best epoch for MS Val "+str(bestEpoch), file=f)

        
        ctdTestResults,ibsTestResults=EvaluateMSforAAE(self.dataLocationTest,weightDir,self.gpu,self.config["cancer_type"],evaluateMulti=False)
        self.writer.add_scalar("TestingCtD"+self.config["cancLabel"], ctdTestResults, bestEpoch)
        self.writer.add_scalar("TestingIBS"+self.config["cancLabel"], ibsTestResults, bestEpoch)
        self.writer.flush()
        self.writer.close()
        
        return bestCtdMS,correspondingIBS

