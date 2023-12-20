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


from torch.utils.tensorboard import SummaryWriter


from PhasedAAE.PhasedAAEModels import Risk_Predictor,MRNAEncoder,MRNADecoder,Adv_Classifier,FC,DNADecoder
from multisurv.src.loss import Loss as MultisurvLoss
from pycox.evaluation import EvalSurv
from DNA_mRNA.TestReconVAE import getTestReconLoss
from DNA_mRNA.TestUnimodalVAE import getTestResultsUnimodal
import pickle
from ray import tune
from trainingHelperFunctions import accuracy,_predictions_to_pycox,compute_KL_loss

torch.manual_seed(42)


#============ PARSE ARGUMENTS =============






class TrainDNAmRNAVAEClass:
    def __init__(self,config,save_dir,save_freq,max_epochs,missingPercent,dataloaderDir,use_gpu,modelLocation=None,modelLocEpoch=-1,fixedConfig=False):
        
        self.fixedConfig=fixedConfig
        folderName="lrAE_{}_lrD_{}_bs_{}_beta_{}_alpha_{}_lamb_{}".format(config["lrAE"],config["lrD"],config["batch_size"],config["beta"],config["alpha"],config["lamb"])
        save_dir=os.path.join(save_dir,folderName)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir,"modelfiles"), exist_ok=True)
        os.makedirs(os.path.join(save_dir,"tensorboardlogs"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir,"tensorboardlogs"))
        
        
        self.netEncA = FC(5000, config["latent_dims"], 5, scaling_factor=2,VAE=True) 
        self.netDecA = DNADecoder(inFeatures=512) 
        self.netEncB = MRNAEncoder(in_features=1000,latentdim=config["latent_dims"],simple=True,VAE=True)#MRNAEncoder(in_features=1000,latentdim=config["latent_dims"])
        self.netDecB = MRNADecoder(inFeatures=512,outFeatures=1000) #MRNADecoder(outFeatures=1000)
        self.netClf = Adv_Classifier(nz=512,n_hidden1=256,n_hidden2=128,n_out=2)#Adv_Classifier(nz=30,n_hidden=60,n_out=2)
        self.netCondClf = Risk_Predictor(nz=512,n_out=30) #Risk_Predictor(nz=30,n_out=30)
        
        if modelLocation is not None:
            self.netEncA.load_state_dict(torch.load(os.path.join(modelLocation,"netEncA_%s.pth" % modelLocEpoch)))
            self.netDecA.load_state_dict(torch.load(os.path.join(modelLocation,"netDecA_%s.pth" % modelLocEpoch)))
            self.netEncB.load_state_dict(torch.load(os.path.join(modelLocation,"netEncB_%s.pth" % modelLocEpoch)))
            self.netDecB.load_state_dict(torch.load(os.path.join(modelLocation,"netDecB_%s.pth" % modelLocEpoch)))
            self.netClf.load_state_dict(torch.load(os.path.join(modelLocation,"netClf_%s.pth" % modelLocEpoch)))
            self.netCondClf.load_state_dict(torch.load(os.path.join(modelLocation,"netCondClf_%s.pth" % modelLocEpoch)))
            print("LOADED PRETRAINED MODEL")
        
        self.output_intervals=torch.arange(0., 365 * 31, 365).double() #NEEDED for computing the multisurv loss
        
       
        
        
        if use_gpu:
            self.netEncA.cuda()
            self.netDecA.cuda()
            self.netEncB.cuda()
            self.netDecB.cuda()
            self.netClf.cuda()
            self.output_intervals.cuda()
            self.netCondClf.cuda()
             
        
        
        # setup optimizer
        self.opt_netEncA = optim.Adam(list(self.netEncA.parameters()), lr=config["lrAE"])
        self.opt_netEncB = optim.Adam(list(self.netEncB.parameters()), lr=config["lrAE"])
        #self.schedulerEnc = ReduceLROnPlateau(self.opt_netEnc, 'max', factor=0.5, verbose=True, threshold=.001)
        
        self.opt_netDecA = optim.Adam(list(self.netDecA.parameters()), lr=config["lrAE"])
        self.opt_netDecB = optim.Adam(list(self.netDecB.parameters()), lr=config["lrAE"])
        #self.schedulerDec = ReduceLROnPlateau(self.opt_netDec, 'max', factor=0.5, verbose=True, threshold=.001)
        
        
        self.opt_netCondClf = optim.Adam(list(self.netCondClf.parameters()), lr=config["lrAE"])
        #self.schedulerCondClf = ReduceLROnPlateau(self.opt_netCondClf, 'max', factor=0.5, verbose=True, threshold=.001)
       
        self.opt_netClf = optim.Adam(list(self.netClf.parameters()), lr=config["lrD"])
        
        # loss criteria
        self.criterion_MultiSurv = MultisurvLoss()
        self.criterion_reconstruct = nn.MSELoss()
        self.criterion_gan = nn.MSELoss()
        
        self.config=config
        self.save_freq=save_freq
        self.missingPercent=missingPercent
        self.dataloaderDir=dataloaderDir
        self.use_gpu=use_gpu
        self.max_epochs=max_epochs
        self.save_dir=save_dir
        
        # setup logger
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            print(config, file=f)
            print(self.netEncA, file=f)
            print(self.netDecA, file=f)
            print(self.netEncB, file=f)
            print(self.netDecB, file=f)
            print(self.netClf, file=f)
            print(self.netCondClf, file=f)
                
        

    def train_AutoencoderA(self,thisBatch):
 
        self.netEncA.train()
        self.netDecA.train()
        self.netCondClf.train()
        
        (data,time,event,*rest)=thisBatch
        
        dataA=data["DNAm"]
        dataA = Variable(dataA)
        dataInput=dataA
        
        if self.use_gpu:
            dataInput= dataInput.cuda()
            #time, event = time.cuda(), event.cuda()
            thisDevice='cuda:0'
        else:
            thisDevice='cpu'
    
        # reset parameter gradients
        self.netEncA.zero_grad() 
        self.netDecA.zero_grad() 
        self.netCondClf.zero_grad()
        
    
        # forward pass
        latents,mu,logvar=self.netEncA(dataInput)
       
        
        recon=self.netDecA(latents)
        
        risk = self.netCondClf(latents)
        clf_class_loss = self.criterion_MultiSurv(risk=risk, times=time, events=event, breaks=self.output_intervals,device=thisDevice)
        
        recon_loss = self.criterion_reconstruct(dataInput, recon)
        kl_loss = self.config["lamb"]*compute_KL_loss(mu, logvar)
        loss = self.config["beta"]*clf_class_loss+self.config["alpha"]*recon_loss+kl_loss
    
        # backpropagate and update model
        loss.backward()
        self.opt_netEncA.step()
        self.opt_netDecA.step()
        self.opt_netCondClf.step()

      
        summary_stats = {'clf_class_loss': clf_class_loss.item()*time.shape[0]}
        summary_stats['risk'] = risk.detach()
        summary_stats['reconLoss'] = recon_loss.detach()*time.shape[0]
    
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
       
        
    
        if self.use_gpu:
            dataInput = dataInput.cuda()
            thisDevice='cuda:0'
        else:
            thisDevice='cpu'
    
        # reset parameter gradients
        self.netEncB.zero_grad() 
        self.netDecB.zero_grad() 
        self.netCondClf.zero_grad()
        
    
        # forward pass
        latents,mu,logvar=self.netEncB(dataInput)
       
        
        
        recon=self.netDecB(latents)
        
        risk = self.netCondClf(latents)
        clf_class_loss = self.criterion_MultiSurv(risk=risk, times=time, events=event, breaks=self.output_intervals,device=thisDevice)
        
        recon_loss = self.criterion_reconstruct(dataInput, recon)
        kl_loss = self.config["lamb"]*compute_KL_loss(mu, logvar)
        loss = self.config["beta"]*clf_class_loss+self.config["alpha"]*recon_loss+kl_loss
    
        # backpropagate and update model
        loss.backward()
        self.opt_netEncB.step()
        self.opt_netDecB.step()
        self.opt_netCondClf.step()

      
        summary_stats = {'clf_class_loss': clf_class_loss.item()*time.shape[0]}
        summary_stats['risk'] = risk.detach()
        summary_stats['reconLoss'] = recon_loss.detach()*time.shape[0]
    
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
    
        if self.use_gpu:
            dataA = dataA.cuda()
            dataB=dataB.cuda()
        
        # reset parameter gradients
        self.netClf.zero_grad()
        
        
        # forward pass
        Alatents,_,_=self.netEncA(dataA)
        Alatents=Alatents.clone().detach()
        AScores = self.netClf(Alatents)
        ALabelsGan = torch.ones(AScores.size(0),2)#.long()
        ALabels = torch.ones(AScores.size(0),).long()
        
        Blatents,_,_=self.netEncB(dataB)
        Blatents=Blatents.clone().detach()
        BScores = self.netClf(Blatents)
        BLabelsGan = torch.zeros(BScores.size(0),2)#.long()
        BLabels = torch.zeros(BScores.size(0),).long()
        if self.use_gpu:
            ALabels = ALabels.cuda()
            BLabels = BLabels.cuda()
            ALabelsGan = ALabelsGan.cuda()
            BLabelsGan = BLabelsGan.cuda()
        
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
        
    
        if self.use_gpu:
            dataA = dataA.cuda()
            dataB=dataB.cuda()
        
        # reset parameter gradients
        self.netEncA.zero_grad()
        self.netEncB.zero_grad()
    
        # forward pass
        Alatents,_,_=self.netEncA(dataA)
        AScores = self.netClf(Alatents)
        
        
        Blatents,_,_=self.netEncB(dataB)
        BScores = self.netClf(Blatents)
        
        ALabelsGan = torch.ones(AScores.size(0),2)#.long()
        BLabelsGan = torch.zeros(BScores.size(0),2)#.long()
        if self.use_gpu:
            ALabelsGan = ALabelsGan.cuda()
            BLabelsGan = BLabelsGan.cuda()
        
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
        else:
            print("YOU MUST SPECIFY A DATA DIRECTORY OR DATALOADER FILE PATH")
            sys.exit()
        ### main training loop
        for epoch in range(self.max_epochs):
            print(epoch)
        
            
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
            
            
            running_durationsB = torch.FloatTensor()
            running_censorsB = torch.LongTensor()
            running_risksB = torch.FloatTensor()
            
            

            for idx, batch in enumerate(dataloaderPairedTrain):
                (_,time,event,*rest)=batch
                
                n_A += time.shape[0]
                n_B += time.shape[0]
      
                out = self.train_AutoencoderA(batch)
                clf_class_lossA += out['clf_class_loss']
                reconLossA += out['reconLoss']
                running_durationsA = torch.cat((running_durationsA,time.data.float()))
                running_censorsA = torch.cat((running_censorsA,event.long().data))
                running_risksA = torch.cat((running_risksA, out["risk"].cpu()))
                
                out = self.train_AutoencoderB(batch)
                clf_class_lossB += out['clf_class_loss']
                reconLossB += out['reconLoss']
                running_durationsB = torch.cat((running_durationsB,time.data.float()))
                running_censorsB = torch.cat((running_censorsB,event.long().data))
                running_risksB = torch.cat((running_risksB, out["risk"].cpu()))
            
                out = self.trainDiscriminator(batch)
                discLoss += out['clf_loss']
                nA_correctDisc += out['accuracyA']
                nB_correctDisc += out['accuracyB']
            
                out = self.trainEncoders(batch)
                generatorClfLoss += out['clf_loss_Encoder']
                
                
            
    
    
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
            
            
            reconLossB /= n_B
            clf_class_lossB /= n_B
            
            generatorClfLoss /= (n_A+n_B)
            
            discLoss /= (n_A+n_B)
                
            _, _, _, _, _, _, _, concordValA=getTestResultsUnimodal(self.netEncA,self.netCondClf,dataloaderPairedVal,self.save_dir,modality="DNAm",use_gpu=self.use_gpu,makeTSNEIdx=-1,perplexity=2)
            _, _, _, _, _, _, _, concordValB=getTestResultsUnimodal(self.netEncB,self.netCondClf,dataloaderPairedVal,self.save_dir,modality="mRNA",use_gpu=self.use_gpu,makeTSNEIdx=-1,perplexity=2)
            _, _, reconLossAVal=getTestReconLoss(self.netEncA,self.netDecA,dataloaderPairedVal,self.save_dir,modality="DNAm",use_gpu=self.use_gpu)
            _, _, reconLossBVal=getTestReconLoss(self.netEncB,self.netDecB,dataloaderPairedVal,self.save_dir,modality="mRNA",use_gpu=self.use_gpu)
            
            
            #self.schedulerEnc.step(epoch_concordAllVal)
            #self.schedulerDec.step(epoch_concordAllVal)
            #self.schedulerCondClf.step(epoch_concordAllVal)
            
           
          
            
            
            
            self.writer.add_scalar("clf_class_lossA", clf_class_lossA, epoch)
            self.writer.add_scalar("clf_class_lossB", clf_class_lossB, epoch)
            
            #self.writer.add_scalar("Clf_Class_Loss_ValUnpairedClin", clf_class_lossAVal, epoch)
            #self.writer.add_scalar("Clf_Class_Loss_ValAll", clf_class_lossTotalVal, epoch)
            
            self.writer.add_scalar("concordValA", concordValA, epoch)
            self.writer.add_scalar("concordValB", concordValB, epoch)
            self.writer.add_scalar("concordATrain", epoch_concordA, epoch)
            self.writer.add_scalar("concordBTrain", epoch_concordB, epoch)
    
            
            self.writer.add_scalar("generatorClfLoss", generatorClfLoss, epoch)
            self.writer.add_scalar("discLoss", discLoss, epoch)
            
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
            if epoch % self.save_freq == 0:
                torch.save(self.netEncA.cpu().state_dict(), os.path.join(self.save_dir,"modelfiles","netEncA_%s.pth" % epoch))
                torch.save(self.netEncB.cpu().state_dict(), os.path.join(self.save_dir,"modelfiles","netEncB_%s.pth" % epoch))
                torch.save(self.netCondClf.cpu().state_dict(), os.path.join(self.save_dir,"modelfiles","netCondClf_%s.pth" % epoch))
                torch.save(self.netDecA.cpu().state_dict(), os.path.join(self.save_dir,"modelfiles","netDecA_%s.pth" % epoch))
                torch.save(self.netDecB.cpu().state_dict(), os.path.join(self.save_dir,"modelfiles","netDecB_%s.pth" % epoch))
                torch.save(self.netClf.cpu().state_dict(), os.path.join(self.save_dir,"modelfiles","netClf_%s.pth" % epoch))
                
                
            if self.use_gpu:
                self.netEncA.cuda()
                self.netDecA.cuda()
                self.netEncB.cuda()
                self.netDecB.cuda()
                self.netClf.cuda()
                self.netCondClf.cuda()
            if not self.fixedConfig:
                tune.report(clf_class_lossATrain=clf_class_lossA,clf_class_lossBTrain=clf_class_lossB,concordValA=concordValA,
                            concordValB=concordValB,concordATrain=epoch_concordA,concordBTrain=epoch_concordB,generatorClfLoss=generatorClfLoss,discLoss=discLoss)
            
        self.writer.flush()
        self.writer.close()

