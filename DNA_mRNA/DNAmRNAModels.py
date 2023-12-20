#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:54:46 2023

@author: sara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:59:01 2022

@author: sara
"""
from bisect import bisect_left
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from typing import Optional

class MRNAEncoder(nn.Module):
    def __init__(self,  latentdim, VAE=True):
        super(MRNAEncoder, self).__init__()
        
        
        self.VAE=VAE
 
        self.fc = nn.Sequential(
                
                torch.nn.Linear(in_features=1000,out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=500, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=latentdim, bias=True),
                torch.nn.ReLU(inplace=True),
                )
        
    
        if self.VAE:
            self.muLayer = nn.Linear(latentdim, latentdim)
            self.logVLayer = nn.Linear(latentdim, latentdim)
        
 
    
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        gpu=std.get_device()
        eps = torch.FloatTensor(std.size()).normal_()
        if gpu>-1:
            device=torch.device('cuda:'+str(gpu))
            eps=eps.to(device)
     
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def encode(self, x):
        h = self.fc(x)
        
        if self.VAE:
            return self.muLayer(h), self.logVLayer(h)
        else:
            return h

 
 
    def forward(self, x):
        if self.VAE:
            mu, logvar = self.encode(x)
            z = self.reparametrize(mu, logvar)
            return z,mu,logvar
        else:
            z = self.encode(x)
            return z



class MRNAEncoderComplex(nn.Module):
    def __init__(self,  latentdim, VAE=True):
        super(MRNAEncoderComplex, self).__init__()
        
        
        self.VAE=VAE
 
        self.fc = nn.Sequential(
                
                torch.nn.Linear(in_features=1000,out_features=800),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=800, out_features=700, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=700, out_features=600, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=600, out_features=600, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=600, out_features=500, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=256, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=256, out_features=latentdim, bias=True),
                torch.nn.ReLU(inplace=True),
                )
        
    
        if self.VAE:
            self.muLayer = nn.Linear(latentdim, latentdim)
            self.logVLayer = nn.Linear(latentdim, latentdim)
        
 
    
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        gpu=std.get_device()
        eps = torch.FloatTensor(std.size()).normal_()
        if gpu>-1:
            device=torch.device('cuda:'+str(gpu))
            eps=eps.to(device)
     
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def encode(self, x):
        h = self.fc(x)
        
        if self.VAE:
            return self.muLayer(h), self.logVLayer(h)
        else:
            return h

 
 
    def forward(self, x):
        if self.VAE:
            mu, logvar = self.encode(x)
            z = self.reparametrize(mu, logvar)
            return z,mu,logvar
        else:
            z = self.encode(x)
            return z




class MRNADecoder(nn.Module):
 
    def __init__(self,latentdim):
        super(MRNADecoder, self).__init__()
        # Embedding layer
        

        self.decoderLayers = []
        self.decoderLayers.append(nn.Linear(latentdim, 512))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(512, 512))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(512, 1000))
        self.decoderLayers.append(nn.ReLU(inplace=True))
    
        self.decoderLayers=nn.Sequential(*self.decoderLayers)
        
    
    def forward(self, z):        
        recon = self.decoderLayers(z)
        return recon
    
class MRNADecoderComplex(nn.Module):
 
    def __init__(self,latentdim):
        super(MRNADecoderComplex, self).__init__()
        # Embedding layer
        

        self.decoderLayers = []
        self.decoderLayers.append(nn.Linear(latentdim, 300))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(300, 400))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(400, 512))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(512, 600))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(600, 700))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(700, 850))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(850, 1000))
        self.decoderLayers.append(nn.ReLU(inplace=True))
    
        self.decoderLayers=nn.Sequential(*self.decoderLayers)
        
    
    def forward(self, z):        
        recon = self.decoderLayers(z)
        return recon
    

class Risk_Predictor(nn.Module):
    """Latent space to risk calculation"""
    def __init__(self, nz, n_out=30):
        super(Risk_Predictor, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=nz,out_features=n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.risk_layer(x)


class Adv_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_hidden1=128, n_hidden2=64,n_out=2):
        super(Adv_Classifier, self).__init__()
        self.nz = nz
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden1),
            nn.ReLU(inplace=True),
#            nn.Linear(n_hidden, n_hidden),
#            nn.ReLU(inplace=True),
 #           nn.Linear(n_hidden, n_hidden),
 #           nn.ReLU(inplace=True),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden2,n_out)
        )

    def forward(self, x):
        return self.net(x)
    


 

class DNAEncoder(nn.Module):
    def __init__(self,  latentdim, VAE=True):
        super(DNAEncoder, self).__init__()
        
        
        self.VAE=VAE
 
        self.fc = nn.Sequential(
                
                torch.nn.Linear(in_features=5000,out_features=3096),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=3096, out_features=2048, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=1024, out_features=512, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=512, out_features=latentdim, bias=True),
                torch.nn.ReLU(inplace=True),
                )
        
        # self.fc = nn.Sequential(
                
        #         torch.nn.Linear(in_features=5000,out_features=2048),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=2048, out_features=512, bias=True),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=512, out_features=latentdim, bias=True),
        #         torch.nn.ReLU(inplace=True),

        #         )
        
    
        if self.VAE:
            self.muLayer = nn.Linear(latentdim, latentdim)
            self.logVLayer = nn.Linear(latentdim, latentdim)
        
 
    
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        gpu=std.get_device()
        eps = torch.FloatTensor(std.size()).normal_()
        if gpu>-1:
            device=torch.device('cuda:'+str(gpu))
            eps=eps.to(device)
     
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def encode(self, x):
        h = self.fc(x)
        
        if self.VAE:
            return self.muLayer(h), self.logVLayer(h)
        else:
            return h

 
 
    def forward(self, x):
        if self.VAE:
            mu, logvar = self.encode(x)
            z = self.reparametrize(mu, logvar)
            return z,mu,logvar
        else:
            z = self.encode(x)
            return z
        
        
class DNADecoder(nn.Module):
 
    def __init__(self,latentdim):
        super(DNADecoder, self).__init__()
        # Embedding layer
        

        self.decoderLayers = []
        self.decoderLayers.append(nn.Linear(latentdim, 512))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(512, 1024))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(1024, 2048))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(2048, 3096))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers.append(nn.Linear(3096, 5000))
        self.decoderLayers.append(nn.ReLU(inplace=True))
        self.decoderLayers=nn.Sequential(*self.decoderLayers)
        
    
    def forward(self, z):        
        recon = self.decoderLayers(z)
        return recon


    