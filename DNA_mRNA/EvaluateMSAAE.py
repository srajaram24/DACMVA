#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:21:38 2023

@author: sara
"""



import sys
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch

from multisurv.src import getPatIds

from multisurvOrig.src import dataset
from multisurvOrig.src.model import Model
from multisurvOrig.src import utils
from multisurvOrig.src import evaluation
import argparse
import itertools

def get_patients_with(cancer_type, dataLocation, split_group='test'):
    labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
    cancer_labels = labels[labels['project_id'] == cancer_type]
    group_cancer_labels = cancer_labels[cancer_labels['group'] == split_group]

    return list(group_cancer_labels['submitter_id'])

def format_bootstrap_output(evaluator):
    results = evaluator.format_results()
    
    for metric in results:
        results[metric] = results[metric].split(' ')
        val = results[metric][0]
        ci_low, ci_high = results[metric][1].split('(')[1].split(')')[0].split('-')
        results[metric] = val, ci_low, ci_high
        results[metric] = [float(x) for x in results[metric]]
    
    return results

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
   
    options.add_argument('-dataLocation', '--dataLocation', action="store", dest="dataLocation",default=None)
    options.add_argument('-weightDir', '--weightDir', action="store", dest="weightDir",default=None)
    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
    options.add_argument('-cancerType', '--cancerType', action="store", dest="cancerType",default=['All'],type=list)
    return options.parse_args()

def create_appendline(cancName,results,saveDir):
    formattedResults = format_bootstrap_output(results)
    ctdVar=formattedResults['Ctd'][2]-formattedResults['Ctd'][1]
    ctdVar=str(round(ctdVar,3))
    ibsVar=formattedResults['IBS'][2]-formattedResults['IBS'][1]
    ibsVar=str(round(ibsVar,3))
    ctd = str(round(results.c_index_td, 3))
    ibs = str(round(results.ibs, 3)) 

    message = cancName+","+ctd+","+ctdVar+","+ibs+","+ibsVar
    with open(os.path.join(saveDir,'evalcancers.txt'), 'a') as f:
        print(message,file=f)

def EvaluateMSforAAE(dataLocation,weightDir,gpu,cancerType=['All'],evaluateMulti=False):
    
    modalities=['mRNA']
    if not isinstance(cancerType,list):
        ValueError("EVALUATEMS expects list for cancer type")
    
    if gpu>-1:
        device = torch.device('cuda:'+str(gpu))
    else:
        device=torch.device("cpu")
    if cancerType[0]=='All':
        cancer_types= ['BLCA','BRCA','CESC','COAD','GBM','HNSC','KIRC','KIRP','LGG','LIHC','LUAD','LUSC','OV','PRAD','SARC','SKCM','STAD','THCA','UCEC']
        exclude_patients = []
    else:
        cancer_types=cancerType
        labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
        exclude_patients = list(labels.loc[~labels['project_id'].isin(cancerType), 'submitter_id'])
    
    with open(os.path.join(weightDir,'evalcancers.txt'),'w') as f:
        #print('-' * 40,file=f)
        print('patients excluded = '+str(len(exclude_patients)),file=f)
        print('cancer types = '+str(cancerType),file=f)
        print('Cancer,Ctd,CtDErr,IBS,IBSErr',file=f)
    datasets=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),modalities=modalities, batch_size=16,exclude_patients=exclude_patients,return_patient_id=True)
    dataloaders = {'test': torch.utils.data.DataLoader(
        datasets['test'], batch_size=16,
        shuffle=True, num_workers=4, drop_last=True)}
    multisurv = Model(dataloaders=dataloaders, device=device, fcBlock=False)
    multisurv.model.mRNA_submodel.load_state_dict(torch.load(os.path.join(weightDir,"netMSEnc.pth")))
    msClfModel=torch.load(os.path.join(weightDir,"netMSClf.pth"))
    if 'risk_layer.0.weight' in msClfModel.keys():
        msClfModel['0.weight'] = msClfModel['risk_layer.0.weight']
        del msClfModel['risk_layer.0.weight']
        msClfModel['0.bias'] = msClfModel['risk_layer.0.bias']
        del msClfModel['risk_layer.0.bias']
    multisurv.model.risk_layer.load_state_dict(msClfModel)
    multisurv.model.eval()
    results = {}
    minimum_n_patients = 20
    labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
    #cancer_types = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t').project_id.unique()
 
        
    allResults = evaluation.Evaluation(model=multisurv, dataset=datasets['test'], device=device)
    allResults.run_bootstrap()
    ctdAllResults = round(allResults.c_index_td, 3)
    ibsAllResults = round(allResults.ibs, 3)
    if  len(cancerType)==1:
        cancLabel=cancerType[0]
    else:
        cancLabel="Multi"
    create_appendline(cancLabel,allResults,weightDir)

    
    if cancerType[0]!="All" and len(cancerType)==1:
        return ctdAllResults,ibsAllResults
    elif not evaluateMulti: 
        return ctdAllResults,ibsAllResults
    else:
        for i, cancer_type in enumerate(cancer_types):
            print('-' * 44)
            print(' ' * 17, f'{i + 1}.', cancer_type)
            print('-' * 44)
        
            patients = get_patients_with(cancer_type,dataLocation)
            if len(patients) < minimum_n_patients:
                continue
            
        #     exclude_patients = [p for p in dataloaders['test'].dataset.patient_ids
        #                         if not p in patients]
        
            thisCancer=[cancer_type]
            exclude_patients = list(labels.loc[~labels['project_id'].isin(thisCancer), 'submitter_id'])
        
            datasets = utils.get_dataloaders(
                data_location=dataLocation,
                labels_file=os.path.join(dataLocation,'labels.tsv'),
                modalities=modalities,
                exclude_patients=exclude_patients,
                batch_size=16,
                return_patient_id=True)
            
        
            data=datasets['test']
        
            results[cancer_type] = evaluation.Evaluation(model=multisurv, dataset=data, device=device)
            results[cancer_type].run_bootstrap()
            print()
        print()
        print()
        
    
        formatted_results = {}
    
            
        for cancer_type in sorted(list(cancer_types)):
            patients = get_patients_with(cancer_type,dataLocation)
            if len(patients) > minimum_n_patients:
                #selected_cancer_types.append(cancer_type)
                formatted_results[cancer_type] = format_bootstrap_output(results[cancer_type])
                
                create_appendline(cancer_type,results[cancer_type],weightDir)
    
    
        with open(os.path.join(weightDir,'evalcancers.txt'),'a') as f:
            print(formatted_results,file=f)
        
        return ctdAllResults,ibsAllResults
        
if __name__ == '__main__':
    if os.getcwd() not in sys.path:
        sys.path.append(os.path.join(os.getcwd()))
    args = setup_args()
    EvaluateMSforAAE(args.dataLocation,args.weightDir,gpu=args.gpu,cancerType=args.cancerType)
    
    