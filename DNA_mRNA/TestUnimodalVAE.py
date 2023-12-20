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

import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.cm as cm
import numpy as np
from warnings import simplefilter
from multisurv.src.loss import Loss as MultisurvLoss
import pandas as pd
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt

import sys


simplefilter(action='ignore', category=FutureWarning)

colors = cm.rainbow(np.linspace(0, 1, 10))
markers = matplotlib.markers.MarkerStyle.filled_markers



def makeTSNEPlotsBoth(tsneBoth,labelsBoth,lastAIdx,name,epIdx,figName,figTitle,save_dir):
    fig, ax = plt.subplots(figsize=(16,10))
    for mod in range(0,2):
        if mod==0:
            tsneVals=tsneBoth[:lastAIdx]
            labels=labelsBoth[:lastAIdx]
            labelLegend='A'
            edge=None
        else:
            tsneVals=tsneBoth[lastAIdx:]
            labels=labelsBoth[lastAIdx:]
            labelLegend='B'
            edge=(0.0, 0.0, 0.0, 1)
        for iclass in range(0, 10):
            idxs = labels==iclass
            if mod==0:
                colorPoints=colors[iclass]
            else:
                colorPoints=(1.0, 1.0, 1.0, 0)
    
            ax.scatter(tsneVals[idxs, 0],
                         tsneVals[idxs, 1],
                         marker=markers[iclass],
                         color=colorPoints,
                         edgecolor=edge,
                         label=r'$%i$'%iclass+labelLegend)
        
    
    ax.set_title("Latent "+name+" at Epoch %s" % epIdx+", "+figTitle, fontsize=24)
    ax.set_xlabel(r'$X^{\mathrm{tSNE}}_1$', fontsize=18)
    ax.set_ylabel(r'$X^{\mathrm{tSNE}}_2$', fontsize=18)
    plt.legend(title=r'Class', loc='best', numpoints=1, fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir,"testresults","tsne",name+"%s" % epIdx+figName))
    
def makeTSNEPlots(tsneEnc,labels,name,epIdx,figName,figTitle,save_dir):
    fig, ax = plt.subplots(figsize=(16,10))
    for iclass in range(0, 10):
        idxs = labels==iclass

        ax.scatter(tsneEnc[idxs, 0],
                     tsneEnc[idxs, 1],
                     marker=markers[iclass],
                     color=colors[iclass],
                     edgecolor=None,
                     label=r'$%i$'%iclass)
    
    
    ax.set_title("Latent "+name+" at Epoch %s" % epIdx+", "+figTitle, fontsize=24)
    ax.set_xlabel(r'$X^{\mathrm{tSNE}}_1$', fontsize=18)
    ax.set_ylabel(r'$X^{\mathrm{tSNE}}_2$', fontsize=18)
    plt.legend(title=r'Class', loc='best', numpoints=1, fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir,"testresults","tsne",name+"%s" % epIdx+figName))

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
    
    dataInput=data[modality]
    dataInput = Variable(dataInput)
   
    
    output_intervals=torch.arange(0., 365 * 31, 365).double()
    
    if thisDevice != "cpu":
        dataInput=dataInput.cuda()
        output_intervals=output_intervals.cuda()
    # forward pass
    with torch.no_grad():
        latents,_,_=netEnc(dataInput)
        
        risk = netCondClf(latents)
        

    # compute losses
    clf_class_loss = criterion_MultiSurv(risk=risk, times=time, events=event, breaks=output_intervals,device=thisDevice)
    
    return risk,clf_class_loss,latents



def embeddings_to_pandas(embeddings,patientLabels):
    dataLocation='/Users/sara/Desktop/multisurvAll/data'
    labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
#     clinical_data = pd.read_csv('../data/clinical_data.tsv', sep='\t',
#                                 na_values=['not reported', 'Not Reported'])

    embeddings = pd.DataFrame(embeddings, columns=['x', 'y'])
    embeddings['submitter_id'] = patientLabels
    embeddings = embeddings.merge(labels.iloc[:, :-1])
    embeddings.set_index('submitter_id', inplace=True)
    
    return embeddings


def _despine(ax):
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('none')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    return ax

def scatter_plot(data, cancer_type_cm, despine=True,
                 regions_to_annotate=None, patients_to_annotate=None,
                 offset_map=None, plot_size=(4, 4)):
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(1, 1, 1)

    # Plot all patients
    ax.scatter(data.x, data.y, s=10, color='lightgrey', label='Others')

    # Plot patient subset for each cancer type
    for cancer_type in cancer_type_cm:
        cancer_data = data[data['project_id'] == cancer_type]
        ax.scatter(cancer_data.x, cancer_data.y, s=10,
                   color=cancer_type_cm[cancer_type], label=cancer_type)
        
        # if regions_to_annotate is not None:
        #     _show_patient_ids(cancer_data, ax, intervals=regions_to_annotate,
        #                       color=cancer_type_cm[cancer_type])
        # if patients_to_annotate is not None:
        #     _show_patient_ids(cancer_data, ax, patient_ids=patients_to_annotate,
        #                       annotation_offsets=offset_map, color=cancer_type_cm[cancer_type])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    if despine:
        ax = _despine(ax)
    
    return fig

def getTestResultsUnimodal(netEnc,netCondClf,dataloaderPairedTest,save_dir,modality,use_gpu=False,makeTSNEIdx=-1,perplexity=2,dataloaderUnpairedTest=None):

    # loss criteria
    if netCondClf is None:
        return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    
    criterion_MultiSurv = MultisurvLoss()
    
    if use_gpu:
        thisDevice='cuda:0'
    else:
        thisDevice='cpu'
    
    netEnc.eval()
   
    netCondClf.eval()
    
    
   
    clf_class_lossPaired = 0
    clf_class_lossA = 0

    nPaired = 0
    nUnpaired = 0
    
    running_durationsAll = torch.FloatTensor()
    running_censorsAll = torch.LongTensor()
    running_risksAll = torch.FloatTensor()
    
    
    running_durations = torch.FloatTensor()
    running_censors = torch.LongTensor()
    running_risks = torch.FloatTensor()
    
    allLatents = torch.FloatTensor()
    allPatients=[]
    for idx, batch in enumerate(dataloaderPairedTest):
        (data,*rest)=batch
        
        time=rest[0]
        event=rest[1]
        
        risk,clf_class_loss,latent=getRisks(data,time,event,netEnc,netCondClf,criterion_MultiSurv,thisDevice,modality=modality)
        clf_class_lossPaired += clf_class_loss*time.shape[0]
        nPaired += time.shape[0]

        
        running_durations = torch.cat((running_durations,time.data.float()))
        running_censors = torch.cat((running_censors,event.long().data))
        running_risks = torch.cat((running_risks, risk.cpu()))
        
        running_durationsAll = torch.cat((running_durationsAll,time.data.float()))
        running_censorsAll = torch.cat((running_censorsAll,event.long().data))
        running_risksAll = torch.cat((running_risksAll, risk.cpu()))
        
        allLatents=torch.cat((allLatents,latent.cpu()))
        if len(rest)>2:
            allPatients.append(rest[2])
    
    
    surv_probs = _predictions_to_pycox(running_risks, time_points=None)
    running_durations = running_durations.cpu().numpy()
    running_censors = running_censors.cpu().numpy()
    epoch_concordPaired= EvalSurv(surv_probs, running_durations, running_censors,censor_surv='km').concordance_td('adj_antolini')
   
            
    
    
    if dataloaderUnpairedTest is not None:
        running_durations = torch.FloatTensor()
        running_censors = torch.LongTensor()
        running_risks = torch.FloatTensor()
        for idx, batch in enumerate(dataloaderUnpairedTest):
            (data,*rest)=batch
            time=rest[0]
            event=rest[1]
            
            risk,clf_class_loss,latents=getRisks(data,time,event,netEnc,netCondClf,criterion_MultiSurv,thisDevice,modality=modality)
            clf_class_lossA += clf_class_loss*time.shape[0]
            nUnpaired += time.shape[0]
            
            running_durations = torch.cat((running_durations,time.data.float()))
            running_censors = torch.cat((running_censors,event.long().data))
            running_risks = torch.cat((running_risks, risk.cpu()))
            
            running_durationsAll = torch.cat((running_durationsAll,time.data.float()))
            running_censorsAll = torch.cat((running_censorsAll,event.long().data))
            running_risksAll = torch.cat((running_risksAll, risk.cpu()))
            
            allLatents=torch.cat((allLatents,latents.cpu()))
            if len(rest)>2:
                allPatients.append(rest[2])
    
    
        surv_probs = _predictions_to_pycox(running_risks, time_points=None)
        running_durations = running_durations.cpu().numpy()
        running_censors = running_censors.cpu().numpy()
        epoch_concordUnpaired = EvalSurv(surv_probs, running_durations, running_censors,censor_surv='km').concordance_td('adj_antolini')
      
        clf_class_lossTotal=(clf_class_lossPaired+clf_class_lossA)/(nPaired+nUnpaired)
        clf_class_lossPaired /= nPaired
        clf_class_lossA /= nUnpaired
    
        surv_probsAll = _predictions_to_pycox(running_risksAll, time_points=None)
        running_durationsAll = running_durationsAll.cpu().numpy()
        running_censorsAll = running_censorsAll.cpu().numpy()
        epoch_concordAll = EvalSurv(surv_probsAll, running_durationsAll, running_censorsAll,censor_surv='km').concordance_td('adj_antolini')
    else:
        epoch_concordUnpaired=-1.0
        clf_class_lossA=-1.0
        clf_class_lossTotal=(clf_class_lossPaired)/(nPaired)
        clf_class_lossPaired = clf_class_lossTotal
        epoch_concordAll=epoch_concordPaired
        
    allLatents=allLatents.cpu().numpy()
    

    #Define TSNE
    #Load TSNE
    # if (perplexity < 0):
    #     tsnePaired = TSNE(n_components=2, init='pca', random_state=0)
    #     tsneUnpaired = TSNE(n_components=2, init='pca', random_state=0)
    #     tsneBothFunction = TSNE(n_components=2, init='pca', random_state=0)
    #     figTitle = "PCA Initialization"
    #     figName="pcaInit"
    # else:
    #     tsnePaired = TSNE(n_components=2, perplexity=perplexity, n_iter=400)
    #     tsneUnpaired = TSNE(n_components=2, perplexity=perplexity, n_iter=400)
    #     tsneBothFunction = TSNE(n_components=2, perplexity=perplexity, n_iter=400)
    #     figTitle = "Perplexity = $%d$"%perplexity
    #     figName =  "tsne-plex%i"%perplexity
    
    #tsneDataNeeded=(allPairedLatents,allPatientsPaired,allUnpairedLatents,allPatientsUnpaired)
        
    if makeTSNEIdx>=0:
        tsne_UnpairedEmbeddings = TSNE(n_components=2, perplexity=50, random_state=42, method='exact').fit_transform(allLatents)
        allPatients=np.asarray(allPatients)
        
        embedUnpaired=embeddings_to_pandas(tsne_UnpairedEmbeddings,allPatients)
        
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        cancer_type_colors = {
            'PRAD': default_colors[6],
            'BRCA': default_colors[0],
            'GBM': default_colors[4],
            'OV': default_colors[1],
            'KIRC': default_colors[5]}
        
        
        
        fig = scatter_plot(data=embedUnpaired, cancer_type_cm=cancer_type_colors, despine=False)
        fig.savefig(os.path.join(save_dir,"testresults","tsne"+"Unpaired1"))
        
        
        
        cancer_type_colors = {
            'THCA': default_colors[6],
            'THYM': default_colors[0],
            'UCEC': default_colors[4],
            'UCS': default_colors[1],
            'UVM': default_colors[5]}
        
        
        fig = scatter_plot(data=embedUnpaired, cancer_type_cm=cancer_type_colors, despine=False)
        fig.savefig(os.path.join(save_dir,"testresults","tsne"+"Unpaired2"))
        
       
        
        
        # makeTSNEPlots(tsne_PairedLatent,allALabels,'A',trEpIdx,figName,figTitle)
        # makeTSNEPlots(tsne_UnpairedLatent,allBLabels,'B',trEpIdx,figName,figTitle)
        # makeTSNEPlotsBoth(tsneBoth,np.hstack((allALabels,allBLabels)),len(allALabels),'Both',trEpIdx,figName,figTitle)
    
    return clf_class_lossPaired, epoch_concordPaired, clf_class_lossA, epoch_concordUnpaired, clf_class_lossTotal, epoch_concordAll
    


    
