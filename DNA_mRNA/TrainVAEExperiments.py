

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


from DNA_mRNA.DNAmRNAModels import Risk_Predictor,MRNAEncoder,MRNADecoder,Adv_Classifier,DNAEncoder,DNADecoder,MRNAEncoderComplex,MRNADecoderComplex
from multisurv.src.loss import Loss as MultisurvLoss
from pycox.evaluation import EvalSurv

from TrainUnimodalMS.TestUnimodal import getTestResultsUnimodal
import pickle
from ray import tune
from trainingHelperFunctions import accuracy,_predictions_to_pycox,compute_KL_loss
import math
from DNA_mRNA.EvaluateMSAAE import EvaluateMSforAAE
import random

torch.manual_seed(42)


#============ PARSE ARGUMENTS =============


def get_lamb(currEpoch,maxEpochs,warmup,maxval):
    if currEpoch<warmup:
        return 0.0
    numCycles=6
    cycleLen=int((maxEpochs-warmup)/numCycles)
    schedulePt=(currEpoch-warmup)%cycleLen
    if schedulePt<cycleLen/2:
        lamb=schedulePt/(cycleLen/2)
    else:
        lamb=1.0
    lamb=maxval*lamb
    
    return lamb





def getTestReconLoss(netEnc,netDec,modality,dataloaderPairedTest,criterion_reconstruct,gpu):

    netEnc.eval()
    netDec.eval()
    reconLoss = 0
    nPaired = 0
    
    for idx, batch in enumerate(dataloaderPairedTest):
        (data,*rest)=batch
        time=rest[0]
        nPaired += time.shape[0]
        dataInput=data[modality]
        
    
        if gpu>-1:
            device=torch.device('cuda:'+str(gpu))
            dataInput=dataInput.to(device)
        # forward pass
        with torch.no_grad():
            allEncOutputs=netEnc(dataInput)
            if type(allEncOutputs) is tuple:
            
                latents=allEncOutputs[0]
                xTgt=dataInput
            else:
                latents=allEncOutputs
                xTgt=dataInput
            recon=netDec(latents)
            recon_loss = criterion_reconstruct(xTgt, recon)
            reconLoss += recon_loss*time.shape[0]
            
  
    reconLoss /= nPaired
    return reconLoss

class TrainVAEExperimentsClass:
    def __init__(self,config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,modelLocation=None,modelLocEpoch=-1,fixedConfig=False):
        
        self.fixedConfig=fixedConfig #here MADE KL COEFF 0, turned off disc/gen updates and also MS updates
        self.config=config
        self.config["lambA"]=0.0
        self.config["lambB"]=0.0
        self.maxValA=0.0#0.04
        self.maxValB=0.0#0.0002
        
        folderName="EvenSimplerAE_lambA{}_B{}_bs_{}_lrAE_{}_ld_{}_wm_{}_beta_{}".format(self.maxValA,self.maxValB,config["batch_size"],config["lrAE"],config["latent_dims"],config["warmup"],config["beta"])
        
        #folderName="TrainAdv_SimpleAElambA{}_B{}_bs_{}_lrAE_{}_ld_{}_wm_{}_beta_{}".format(self.maxValA,self.maxValB,config["batch_size"],config["lrAE"],config["latent_dims"],config["warmup"],config["beta"])
        
        #folderName="lrAE_{}_lrD_{}_bs_{}_beta_{}_alpha_{}_gam_{}_lrMS_{}_t_{}_Cyclic01BCE".format(config["lrAE"],config["lrD"],config["batch_size"],config["beta"],config["alpha"],config["gamma"],config["lrMS"],config["t"])
        save_dir=os.path.join(save_dir,folderName)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir,"modelfiles"), exist_ok=True)
        os.makedirs(os.path.join(save_dir,"tensorboardlogs"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir,"tensorboardlogs"))
        
        
        self.netEncA = DNAEncoder(latentdim=config["latent_dims"],VAE=config["VAE"]) 
        self.netDecA = DNADecoder(latentdim=config["latent_dims"]) 
        self.netEncB = MRNAEncoder(latentdim=config["latent_dims"],VAE=config["VAE"])#MRNAEncoder(in_features=1000,latentdim=config["latent_dims"])
        self.netDecB = MRNADecoder(latentdim=config["latent_dims"]) #MRNADecoder(outFeatures=1000)
        self.netClf = Adv_Classifier(nz=config["latent_dims"],n_out=2)#Adv_Classifier(nz=30,n_hidden=60,n_out=2)
        self.netCondClf = Risk_Predictor(nz=config["latent_dims"],n_out=30) #Risk_Predictor(nz=30,n_out=30)
        
        
        self.netMSEnc=MRNAEncoder(latentdim=config["latent_dims"],VAE=False)
        self.netMSClf = Risk_Predictor(nz=config["latent_dims"],n_out=30)
        
        if modelLocation is not None:
            self.netEncA.load_state_dict(torch.load(os.path.join(modelLocation,"netEncA_%s.pth" % modelLocEpoch)))
            self.netDecA.load_state_dict(torch.load(os.path.join(modelLocation,"netDecA_%s.pth" % modelLocEpoch)))
            self.netEncB.load_state_dict(torch.load(os.path.join(modelLocation,"netEncB_%s.pth" % modelLocEpoch)))
            self.netDecB.load_state_dict(torch.load(os.path.join(modelLocation,"netDecB_%s.pth" % modelLocEpoch)))
            self.netClf.load_state_dict(torch.load(os.path.join(modelLocation,"netClf_%s.pth" % modelLocEpoch)))
            self.netCondClf.load_state_dict(torch.load(os.path.join(modelLocation,"netCondClf_%s.pth" % modelLocEpoch)))
            print("LOADED PRETRAINED MODEL")
        
        self.output_intervals=torch.arange(0., 365 * 31, 365).double() #NEEDED for computing the multisurv loss
        
       
        
        if gpu>-1:
            self.device = torch.device('cuda:'+str(gpu))
        else:
            self.device=torch.device("cpu")
            
        if gpu>-1:
            self.netEncA.to(self.device)
            self.netDecA.to(self.device)
            self.netEncB.to(self.device)
            self.netDecB.to(self.device)
            self.netClf.to(self.device)
            self.netMSEnc.to(self.device)
            self.netMSClf.to(self.device)
            self.output_intervals.to(self.device)
            self.netCondClf.to(self.device)
             
        
        
        # setup optimizer
        self.opt_netEncA = optim.Adam(list(self.netEncA.parameters()), lr=config["lrAE"])
        self.opt_netEncB = optim.Adam(list(self.netEncB.parameters()), lr=config["lrAE"])
        self.opt_netMSEnc = optim.Adam(list(self.netMSEnc.parameters()), lr=config["lrMS"])
        self.opt_netMSClf = optim.Adam(list(self.netMSClf.parameters()), lr=config["lrMS"])
        #self.schedulerEnc = ReduceLROnPlateau(self.opt_netEnc, 'max', factor=0.5, verbose=True, threshold=.001)
        
        self.opt_netDecA = optim.Adam(list(self.netDecA.parameters()), lr=config["lrAE"])
        self.opt_netDecB = optim.Adam(list(self.netDecB.parameters()), lr=config["lrAE"])
        #self.schedulerDec = ReduceLROnPlateau(self.opt_netDec, 'max', factor=0.5, verbose=True, threshold=.001)
        
        
        self.opt_netCondClf = optim.Adam(list(self.netCondClf.parameters()), lr=config["lrAE"])
        #self.schedulerCondClf = ReduceLROnPlateau(self.opt_netCondClf, 'max', factor=0.5, verbose=True, threshold=.001)
       
        self.opt_netClf = optim.Adam(list(self.netClf.parameters()), lr=config["lrD"])
        
        # loss criteria
        self.criterion_MultiSurv = MultisurvLoss()
        self.criterion_reconstruct = torch.nn.L1Loss()
        
        self.criterion_gan = nn.CrossEntropyLoss()
        
        
        self.missingPercent=missingPercent
        self.dataloaderDir=dataloaderDir
        self.gpu=gpu
        self.max_epochs=max_epochs
        self.save_dir=save_dir
        self.dataLocationTest=dataLocationTest
        # setup logger
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            print(config, file=f)
            print("missing percent is "+str(self.missingPercent))
            print(self.netEncA, file=f)
            print(self.netDecA, file=f)
            print(self.netEncB, file=f)
            print(self.netDecB, file=f)
            print(self.netClf, file=f)
            print(self.netCondClf, file=f)
            print(self.netMSEnc, file=f)
            print(self.netMSClf, file=f)
                
        

    def train_AutoencoderA(self,thisBatch):
 
        self.netEncA.train()
        self.netDecA.train()
        self.netCondClf.train()
        
        (data,time,event,*rest)=thisBatch
        
        dataA=data["DNAm"]
        dataA = Variable(dataA)
        dataInput=dataA
        
        if self.gpu>-1:
            dataInput= dataInput.to(self.device)
            
    
        # reset parameter gradients
        self.netEncA.zero_grad() 
        self.netDecA.zero_grad() 
        self.netCondClf.zero_grad()
        
    
        # forward pass
        if self.config["VAE"]:
            latents,mu,logvar=self.netEncA(dataInput)
        else:
            latents=self.netEncA(dataInput)
       
        
        recon=self.netDecA(latents)
        
        risk = self.netCondClf(latents)
        clf_class_loss = self.criterion_MultiSurv(risk=risk, times=time, events=event, breaks=self.output_intervals,device=self.device)
        
        recon_loss = self.criterion_reconstruct(dataInput, recon)
        
        loss = self.config["beta"]*clf_class_loss+self.config["alpha"]*recon_loss
        
        if self.config["VAE"]:
            kl_loss = (1/time.shape[0])*compute_KL_loss(mu, logvar)
            loss+=self.config["lambA"]*kl_loss
    
        # backpropagate and update model
        loss.backward()
        self.opt_netEncA.step()
        self.opt_netDecA.step()
        self.opt_netCondClf.step()

      
        summary_stats = {'clf_class_loss': clf_class_loss.item()*time.shape[0]}
        summary_stats['risk'] = risk.detach()
        summary_stats['reconLoss'] = recon_loss.detach()*time.shape[0]
        summary_stats['kl_loss'] = kl_loss.detach()*time.shape[0]
        return summary_stats
    
    
    def train_AutoencoderB(self,thisBatch):
        #Freze the B encoder and decoder and detach so as to train only the A as well as the class classifier
        #Put B autoencoder in eval mode also (affects dropout layers)
        self.netEncB.train()
        self.netDecB.train()
        self.netCondClf.train()
        
        (data,time,event,*rest)=thisBatch
        
        dataB=data["mRNA"]
        dataB = Variable(dataB)
        dataInput=dataB
       
        
    
        if self.gpu>-1:
            dataInput = dataInput.to(self.device)

    
        # reset parameter gradients
        self.netEncB.zero_grad() 
        self.netDecB.zero_grad() 
        self.netCondClf.zero_grad()
        
    
        # forward pass
        if self.config["VAE"]:
            latents,mu,logvar=self.netEncB(dataInput)
        else:
            latents=self.netEncB(dataInput)
       
        
        
        recon=self.netDecB(latents)
        
        risk = self.netCondClf(latents)
        clf_class_loss = self.criterion_MultiSurv(risk=risk, times=time, events=event, breaks=self.output_intervals,device=self.device)
        
        recon_loss = self.criterion_reconstruct(dataInput, recon)
        
        loss = self.config["beta"]*clf_class_loss+self.config["alpha"]*recon_loss
        
        if self.config["VAE"]:
            kl_loss = (1/time.shape[0])*compute_KL_loss(mu, logvar)
            loss+=self.config["lambB"]*kl_loss
    
        # backpropagate and update model
        loss.backward()
        self.opt_netEncB.step()
        self.opt_netDecB.step()
        self.opt_netCondClf.step()

      
        summary_stats = {'clf_class_loss': clf_class_loss.item()*time.shape[0]}
        summary_stats['risk'] = risk.detach()
        summary_stats['reconLoss'] = recon_loss.detach()*time.shape[0]
        summary_stats['kl_loss'] = kl_loss.detach()*time.shape[0]
    
        return summary_stats
    
    def getCMRecon(self,thisBatch):
        
        self.netEncA.eval()
        self.netDecB.eval()
        
        
        (data,time,event,*rest)=thisBatch
        
        dataInputA=data["DNAm"]
        dataInputB=data["mRNA"]
       
        
    
        if self.gpu>-1:
            dataInputA = dataInputA.to(self.device)
            dataInputB = dataInputB.to(self.device)

        with torch.no_grad():
            # forward pass
            if self.config["VAE"]:
                latents,mu,logvar=self.netEncA(dataInputA)
            else:
                latents=self.netEncA(dataInputA)
            recon=self.netDecB(latents)
            recon_lossMAE = self.criterion_reconstruct(dataInputB, recon)
        
            
      
        summary_stats={'reconLossMAE': recon_lossMAE.detach()*time.shape[0]}
       
    
        return summary_stats



    def trainDiscriminator(self,thisBatch):
        
        self.netEncA.eval()
        self.netEncB.eval()
        self.netClf.train()
    
        (data,time,event,*rest)=thisBatch
        
        dataA=data["DNAm"]
        dataA = Variable(dataA)
        dataB=data["mRNA"]
        dataB=Variable(dataB)
    
        if self.gpu>-1:
            dataA = dataA.to(self.device)
            dataB=dataB.to(self.device)
        
        # reset parameter gradients
        self.netClf.zero_grad()
        
        
        # forward pass
        if self.config["VAE"]:
            Alatents,_,_=self.netEncA(dataA)
            Alatents=Alatents.clone().detach()
        else:
            Alatents=self.netEncA(dataA).clone().detach()
        AScores = self.netClf(Alatents)
        ALabelsGan = torch.ones(AScores.size(0),).long()
        ALabels = torch.ones(AScores.size(0),).long()
        
        if self.config["VAE"]:
            Blatents,_,_=self.netEncB(dataB)
            Blatents=Blatents.clone().detach()
        else:
            Blatents=self.netEncB(dataB).clone().detach()
            
        BScores = self.netClf(Blatents)
        BLabelsGan = torch.zeros(BScores.size(0),).long()
        BLabels = torch.zeros(BScores.size(0),).long()
        if self.gpu>-1:
            ALabels = ALabels.to(self.device)
            BLabels = BLabels.to(self.device)
            ALabelsGan = ALabelsGan.to(self.device)
            BLabelsGan = BLabelsGan.to(self.device)
        
        clf_loss=self.criterion_gan(AScores, ALabelsGan)+self.criterion_gan(BScores, BLabelsGan)
        
        
        loss = clf_loss
    
        # backpropagate and update model
        loss.backward()
        self.opt_netClf.step()
        
    
        summary_stats = {'clf_loss': loss*time.shape[0]*2,'accuracyA': accuracy(AScores, ALabels),'accuracyB': accuracy(BScores, BLabels)}
        
        return summary_stats
    
  
    
    
    def trainEncoders(self,thisBatch):
        
        self.netEncA.train()
        self.netEncB.train()
        self.netClf.eval()
        
        for param in self.netClf.parameters():
            param.requires_grad = False
    
        (data,time,event,*rest)=thisBatch
        dataA=data["DNAm"]
        dataA = Variable(dataA)
        dataB=data["mRNA"]
        dataB=Variable(dataB)
        
    
        if self.gpu>-1:
            dataA = dataA.to(self.device)
            dataB=dataB.to(self.device)
        
        # reset parameter gradients
        self.netEncA.zero_grad()
        self.netEncB.zero_grad()
    
        # forward pass
        if self.config["VAE"]:
            Alatents,_,_=self.netEncA(dataA)
        else:
            Alatents=self.netEncA(dataA)
            
        AScores = self.netClf(Alatents)
        
        if self.config["VAE"]:
            Blatents,_,_=self.netEncB(dataB)
        else:
            Blatents=self.netEncB(dataB)
            
        BScores = self.netClf(Blatents)
        
        ALabelsGan = torch.ones(AScores.size(0),).long()
        BLabelsGan = torch.zeros(BScores.size(0),).long()
        if self.gpu>-1:
            ALabelsGan = ALabelsGan.to(self.device)
            BLabelsGan = BLabelsGan.to(self.device)
        
        clf_loss=self.criterion_gan(AScores, BLabelsGan)+self.criterion_gan(BScores, ALabelsGan)
        
        
        loss = clf_loss
    
        # backpropagate and update model
        loss.backward()
        self.opt_netEncA.step()
        self.opt_netEncB.step()
   
        summary_stats = {'clf_loss_Encoder': loss*time.shape[0]*2}
        
        for param in self.netClf.parameters():
            param.requires_grad = True
        
        return summary_stats
    
    def trainMS(self,origBatch,imputedBatch,epochRatio):
        self.netMSEnc.train()
        self.netMSClf.train()
        # reset parameter gradients
        self.netMSEnc.zero_grad() 
        self.netMSClf.zero_grad() 
       
        (data,timeOrig,eventOrig,*rest)=origBatch
        dataOrig=data["mRNA"]
        numOrig=timeOrig.shape[0]
       
        if self.gpu>-1:
            dataOrig = dataOrig.to(self.device)

 
        
        # forward pass
        latentsOrig=self.netMSEnc(dataOrig)
        riskOrig = self.netMSClf(latentsOrig)
        msLossOrig = self.criterion_MultiSurv(risk=riskOrig, times=timeOrig, events=eventOrig, breaks=self.output_intervals,device=self.device)
        
        if len(imputedBatch)>0:
            (dataImp,timeImp,eventImp,*rest)=imputedBatch
          
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
            
            summary_stats = {'msLossOrig': msLossOrig.item()*numOrig}
            summary_stats['msLossImp']=msLossImp.item()*numImp
            summary_stats['numImp']=numImp
            
        else:
            msLossImp=0
            numImp=0
            prob=0
            
            summary_stats = {'msLossOrig': msLossOrig.item()*numOrig}
            summary_stats['msLossImp']=0.0
            summary_stats['numImp']=0
            
        loss = (numOrig*msLossOrig+prob*numImp*msLossImp)/(numOrig+prob*numImp)
    
        # backpropagate and update model
        loss.backward()
        self.opt_netMSEnc.step()
        self.opt_netMSClf.step()
       

    
        return summary_stats
    
    
    def createImputations(self,dataloaderDNATrain):

        self.netEncA.eval()
        self.netDecB.eval()

        self.netMSEnc.eval()
        self.netMSClf.eval()



             
        dnaTrainDataset=dataloaderDNATrain.dataset

        imputedBatch=[]

        numAttempts=0
        while len(imputedBatch)<self.config["batch_size"] and numAttempts<len(dnaTrainDataset):
            numAttempts+=1
            singleSample=random.choice(dnaTrainDataset)
            (dataPat,*rest)=singleSample
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
                latentsImp=self.netMSEnc(imputedB)
                riskImp = self.netMSClf(latentsImp).clone().detach()
                timeTensor=torch.tensor([rest[0]])
                eventTensor=torch.tensor([rest[1]])
                lossVal=self.criterion_MultiSurv(risk=riskImp, times=timeTensor, events=eventTensor, breaks=self.output_intervals,device=self.device)
                if lossVal<self.config["t"]:
                    imputedB=torch.squeeze(imputedB,0)
                    imputedBatch.append((imputedB,*rest))
        
        if len(imputedBatch)>0:
            dl=torch.utils.data.DataLoader(imputedBatch, batch_size=len(imputedBatch),shuffle=True)
            imputedBatch=next(iter(dl))
        else:
            imputedBatch=[]
            
        # if self.use_gpu:
        #     self.netEncA.cuda()
        #     self.netDecB.cuda()
        #     self.netMSEnc.cuda()
        #     self.netMSClf.cuda()
            
        return imputedBatch



    def trainCM(self):
        #additional params
        
        # load data
        

        if self.dataloaderDir is not None:
            with open(os.path.join(self.dataloaderDir,"dataloaderPairedTrain"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                dataloaderPairedTrain = pickle.load(file)
            # with open(os.path.join(self.dataloaderDir,"dataloaderUnpairedmRNATrain"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
            #     dataloaderUnpairedmRNATrain = pickle.load(file)
            # with open(os.path.join(self.dataloaderDir,"dataloaderUnpairedmiRNATrain"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
            #     dataloaderUnpairedmiRNATrain = pickle.load(file)
            with open(os.path.join(self.dataloaderDir,"dataloaderPairedVal"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                dataloaderPairedVal = pickle.load(file)
            # with open(os.path.join(self.dataloaderDir,"dataloaderUnpairedmRNAVal"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
            #     dataloaderUnpairedmRNAVal = pickle.load(file)
            # with open(os.path.join(self.dataloaderDir,"dataloaderUnpairedmiRNAVal"+str(self.config["batch_size"])+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
            #     dataloaderUnpairedmiRNAVal = pickle.load(file)
            with open(os.path.join(self.dataloaderDir,"dataloaderAllRNAVal"+str(32)+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                 dataloaderRNAVal = pickle.load(file)
            
            with open(os.path.join(self.dataloaderDir,"dataloaderAllDNATrain"+str(32)+"mp"+str(int(self.missingPercent*100))), 'rb') as file:
                 dataloaderDNATrain = pickle.load(file)
        else:
            print("YOU MUST SPECIFY A DATA DIRECTORY OR DATALOADER FILE PATH")
            sys.exit()
        ### main training loop
        
        bestConcordMS=0.0
        bestEpoch=-1
        for epoch in range(self.max_epochs):
            print(epoch)
            self.config["lambA"]=get_lamb(epoch,self.max_epochs,self.config["warmup"],self.maxValA)
            self.config["lambB"]=get_lamb(epoch,self.max_epochs,self.config["warmup"],self.maxValB)
            if epoch<self.config["warmup"]:
                self.config["alpha"]=10.0
            else:
                self.config["alpha"]=1.0
            clf_class_lossA = 0
            reconLossA=0
            n_A=0
            
            
            running_durationsA = torch.FloatTensor()
            running_censorsA = torch.LongTensor()
            running_risksA = torch.FloatTensor()
            
            clf_class_lossB = 0
            reconLossB=0
            n_B=0
            
            discLoss=0
            nA_correctDisc=0
            nB_correctDisc=0
            
            generatorClfLoss=0
            msLossOrig=0
            msLossImp=0
            numImp=0
            
            running_durationsB = torch.FloatTensor()
            running_censorsB = torch.LongTensor()
            running_risksB = torch.FloatTensor()
            klA=0.0
            klB=0.0
            
            
            cmReconMAETrain=0.0
            
            cmReconMAEVal=0.0
            reconLossAVal=getTestReconLoss(self.netEncA,self.netDecA,"DNAm",dataloaderPairedVal,self.criterion_reconstruct,self.gpu)
    
            reconLossBVal=getTestReconLoss(self.netEncB,self.netDecB,"mRNA",dataloaderPairedVal,self.criterion_reconstruct,self.gpu)
            
            for idx, batch in enumerate(dataloaderPairedTrain):
                out=self.getCMRecon(batch)
                
                cmReconMAETrain+=out["reconLossMAE"]

            for idx, batch in enumerate(dataloaderPairedTrain):
                (_,time,event,*rest)=batch
                
                n_A += time.shape[0]
                n_B += time.shape[0]
      
                out = self.train_AutoencoderA(batch)
                clf_class_lossA += out['clf_class_loss']
                reconLossA += out['reconLoss']
                klA += out['kl_loss']
                running_durationsA = torch.cat((running_durationsA,time.data.float()))
                running_censorsA = torch.cat((running_censorsA,event.long().data))
                running_risksA = torch.cat((running_risksA, out["risk"].cpu()))
                
                out = self.train_AutoencoderB(batch)
                clf_class_lossB += out['clf_class_loss']
                reconLossB += out['reconLoss']
                klB += out['kl_loss']
                running_durationsB = torch.cat((running_durationsB,time.data.float()))
                running_censorsB = torch.cat((running_censorsB,event.long().data))
                running_risksB = torch.cat((running_risksB, out["risk"].cpu()))
            
                out = self.trainDiscriminator(batch)
                discLoss += out['clf_loss']
                nA_correctDisc += out['accuracyA']
                nB_correctDisc += out['accuracyB']
            
                out = self.trainEncoders(batch)
                generatorClfLoss += out['clf_loss_Encoder']

                
                # # if epoch<30:
                # #     batchImputed=[]
                # # else:
                # if epoch>((self.max_epochs/2)+5):
                #     if self.config["gamma"]==0:
                #         batchImputed=[]
                #     else:
                #         batchImputed=self.createImputations(dataloaderDNATrain)
                #     #epoch/self.max_epochs
                #     out=self.trainMS(batch,batchImputed,-1) #here
                #     msLossImp += out['msLossImp']
                #     msLossOrig += out['msLossOrig']
                #     numImp+=out["numImp"]
                    
                
                
                
            
    
    
            surv_probsA= _predictions_to_pycox(running_risksA, time_points=None)
            running_durationsA = running_durationsA.cpu().numpy()
            running_censorsA = running_censorsA.cpu().numpy()
            epoch_concordA = EvalSurv(surv_probsA, running_durationsA, running_censorsA,censor_surv='km').concordance_td('adj_antolini')
            
    
            
            surv_probsB= _predictions_to_pycox(running_risksB, time_points=None)
            running_durationsB = running_durationsB.cpu().numpy()
            running_censorsB = running_censorsB.cpu().numpy()
            epoch_concordB = EvalSurv(surv_probsB, running_durationsB, running_censorsB,censor_surv='km').concordance_td('adj_antolini')
        
            ADiscAccuracy=nA_correctDisc/n_A
            BDiscAccuracy=nB_correctDisc/n_B
            
            reconLossA /= n_A
            clf_class_lossA /= n_A
            klA /= n_A
            
            
            reconLossB /= n_B
            clf_class_lossB /= n_B
            klB /= n_B
            
            generatorClfLoss /= (n_A+n_B)
            
            discLoss /= (n_A+n_B)
            if numImp>0:
                msLossImp /= numImp
            msLossOrig /= n_A
            
            
            cmReconMAETrain /= n_B
                
            _, _, _, _, _, _, _,concordValA=getTestResultsUnimodal(self.netEncA,self.netCondClf,dataloaderPairedVal,self.save_dir,modality="DNAm",gpu=self.gpu,makeTSNEIdx=-1,perplexity=2)
            _, _, _, _, _, _, _,concordValB=getTestResultsUnimodal(self.netEncB,self.netCondClf,dataloaderPairedVal,self.save_dir,modality="mRNA",gpu=self.gpu,makeTSNEIdx=-1,perplexity=2)
            _, _, _, _, _, _, _,concordValMS=getTestResultsUnimodal(self.netMSEnc,self.netMSClf,dataloaderRNAVal,self.save_dir,modality="mRNA",gpu=self.gpu,makeTSNEIdx=-1,perplexity=2)
    
            
            nPairedVal=0
            for idx, batch in enumerate(dataloaderPairedVal):
                (_,time,event,*rest)=batch
                
                nPairedVal += time.shape[0]
                out=self.getCMRecon(batch)
                
                cmReconMAEVal+=out["reconLossMAE"]
            
            cmReconMAEVal /= nPairedVal
            
            #self.schedulerEnc.step(epoch_concordAllVal)
            #self.schedulerDec.step(epoch_concordAllVal)
            #self.schedulerCondClf.step(epoch_concordAllVal)
            
           
          
            
            
            
            self.writer.add_scalar("clf_class_lossA", clf_class_lossA, epoch)
            self.writer.add_scalar("clf_class_lossB", clf_class_lossB, epoch)
            self.writer.add_scalar("klA", klA, epoch)
            self.writer.add_scalar("klB", klB, epoch)
            self.writer.add_scalar("klWeightA", self.config["lambA"], epoch)
            self.writer.add_scalar("klWeightB", self.config["lambB"], epoch)
         
            self.writer.add_scalar("cmReconMAETrain", cmReconMAETrain, epoch)
           
            self.writer.add_scalar("cmReconMAEVal", cmReconMAEVal, epoch)
            
            #self.writer.add_scalar("Clf_Class_Loss_ValUnpairedClin", clf_class_lossAVal, epoch)
            #self.writer.add_scalar("Clf_Class_Loss_ValAll", clf_class_lossTotalVal, epoch)
            
            self.writer.add_scalar("concordValA", concordValA, epoch)
            self.writer.add_scalar("concordValMS", concordValMS, epoch)
            self.writer.add_scalar("concordValB", concordValB, epoch)
            self.writer.add_scalar("concordATrain", epoch_concordA, epoch)
            self.writer.add_scalar("concordBTrain", epoch_concordB, epoch)
    
            
            self.writer.add_scalar("generatorClfLoss", generatorClfLoss, epoch)
            self.writer.add_scalar("discLoss", discLoss, epoch)
            
            self.writer.add_scalar("msLossImp", msLossImp, epoch)
            self.writer.add_scalar("msLossOrig", msLossOrig, epoch)
            
            
            self.writer.add_scalar("reconLossATrain", reconLossA, epoch)
            self.writer.add_scalar("reconLossBTrain", reconLossB, epoch)
            self.writer.add_scalar("reconLossAVal", reconLossAVal, epoch)
            self.writer.add_scalar("reconLossBVal", reconLossBVal, epoch)
            
            self.writer.add_scalar("ADiscAccuracy", ADiscAccuracy, epoch)
            self.writer.add_scalar("BDiscAccuracy", BDiscAccuracy, epoch)
            
            #self.writer.add_scalar("reconLossPairedVal", reconLossPairedVal, epoch)
            #self.writer.add_scalar("reconLossUnpairedVal", reconLossUnpairedVal, epoch)
            #self.writer.add_scalar("reconLossTotalVal", reconLossTotalVal, epoch)
            
            # save model
            if concordValMS>bestConcordMS: #here
                weightDir=os.path.join(self.save_dir,"modelfiles") # here
                bestConcordMS=concordValMS
                bestEpoch=epoch
                if os.path.exists(weightDir):
                    shutil.rmtree(weightDir)
                os.makedirs(weightDir, exist_ok=True)
                torch.save(self.netEncA.cpu().state_dict(), os.path.join(weightDir,"netEncA.pth"))
                torch.save(self.netEncB.cpu().state_dict(), os.path.join(weightDir,"netEncB.pth"))
                torch.save(self.netCondClf.cpu().state_dict(), os.path.join(weightDir,"netCondClf.pth"))
                torch.save(self.netDecA.cpu().state_dict(), os.path.join(weightDir,"netDecA.pth"))
                torch.save(self.netDecB.cpu().state_dict(), os.path.join(weightDir,"netDecB.pth"))
                torch.save(self.netClf.cpu().state_dict(), os.path.join(weightDir,"netClf.pth"))
                torch.save(self.netMSEnc.cpu().state_dict(), os.path.join(weightDir,"netMSEnc.pth"))
                torch.save(self.netMSClf.cpu().state_dict(), os.path.join(weightDir,"netMSClf.pth"))
                
                
            if self.gpu>-1:
                self.netEncA.to(self.device)
                self.netDecA.to(self.device)
                self.netEncB.to(self.device)
                self.netDecB.to(self.device)
                self.netClf.to(self.device)
                self.netCondClf.to(self.device)
                self.netMSEnc.to(self.device)
                self.netMSClf.to(self.device)
            if not self.fixedConfig:
                tune.report(clf_class_lossATrain=clf_class_lossA,clf_class_lossBTrain=clf_class_lossB,concordValA=concordValA,
                            concordValB=concordValB,concordATrain=epoch_concordA,concordBTrain=epoch_concordB,generatorClfLoss=generatorClfLoss,discLoss=discLoss)
       
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            print("Best epoch for MS Val "+str(bestEpoch), file=f)

        
        ctdAllResults,ibsTestResults=EvaluateMSforAAE(self.dataLocationTest,weightDir,self.gpu)
        self.writer.add_scalar("TestingCtDAll", ctdAllResults, bestEpoch)
        self.writer.add_scalar("TestingIBSAll", ibsTestResults, bestEpoch)
        self.writer.flush()
        self.writer.close()

