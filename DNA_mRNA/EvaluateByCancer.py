#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:20:50 2023

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
    options.add_argument('-lungOnly', '--lungOnly', action="store_true", dest="lungOnly",default=False)
    options.add_argument('-fusion', '--fusion', action="store", dest="fusion",default='max')
    
    options.add_argument('-dataLoaderLocPaired', '--dataLoaderLocPaired', action="store", dest="dataLoaderLocPaired",default=None)
    options.add_argument('-dataLoaderLocUnpairedA', '--dataLoaderLocUnpairedA', action="store", dest="dataLoaderLocUnpairedA",default=None)
    options.add_argument('-dataLoaderLocUnpairedB', '--dataLoaderLocUnpairedB', action="store", dest="dataLoaderLocUnpairedB",default=None)
    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
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

def evaluateByCancerType(dataLocation,weightDir,lungOnly,fusion,dataLoaderLocPaired,dataLoaderLocUnpairedA,dataLoaderLocUnpairedB,gpu):
    
    modalities=['mRNA']
    
    if gpu>-1:
        device = torch.device('cuda:'+str(gpu))
    else:
        device=torch.device("cpu")
    

    if lungOnly:
        cancer_types= ['LUAD', 'LUSC']
        labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
        exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])
    else:
        cancer_types= ['BLCA','BRCA','CESC','COAD','GBM','HNSC','KIRC','KIRP','LGG','LIHC','LUAD','LUSC','OV','PRAD','SARC','SKCM','STAD','THCA','UCEC']
        # labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
        # exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])
        exclude_patients = []
    with open(os.path.join(weightDir,'evalcancers.txt'),'w') as f:
        #print('-' * 40,file=f)
        print('Cancer,Ctd,CtDErr,IBS,IBSErr',file=f)
    datasets=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),modalities=modalities, batch_size=16,exclude_patients=exclude_patients,return_patient_id=True)
    dataloaders = {'train': torch.utils.data.DataLoader(
        datasets['train'], batch_size=16,
        shuffle=True, num_workers=4, drop_last=True),
                    'val': torch.utils.data.DataLoader(
        datasets['val'], batch_size=16,
        shuffle=True, num_workers=4, drop_last=True),
                    'test': torch.utils.data.DataLoader(
        datasets['test'], batch_size=16,
        shuffle=True, num_workers=4, drop_last=True)}
    multisurv = Model(dataloaders=dataloaders, device=device, fusion_method=fusion,fcBlock=False)
    multisurv.load_weights(os.path.join(weightDir,'multisurv.pth'))
    multisurv.model.eval()
    results = {}
    minimum_n_patients = 20
    labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
    #cancer_types = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t').project_id.unique()
    
    allTestPatIDs=[]
    for dataItem in datasets['test']:
        allTestPatIDs.append(dataItem[-1])
        
    allResults = evaluation.Evaluation(model=multisurv, dataset=datasets['test'], device=device)
    allResults.run_bootstrap()
    create_appendline("All",allResults,weightDir)

        
        
    if not lungOnly:

        patIDsPaired,patIDsUnpairedA,patIDsUnpairedB=getPatIds.getPatientIDs(dataLoaderLocPaired,dataLoaderLocUnpairedA,dataLoaderLocUnpairedB)
        
        patIDsinDataloadersTest=list(itertools.chain(*[patIDsPaired,patIDsUnpairedA,patIDsUnpairedB]))
        patIDsMissinginDLTest=list(set(allTestPatIDs)-set(patIDsinDataloadersTest))
        
        excludeForUnpairedA=list(itertools.chain(*[patIDsPaired,patIDsUnpairedB,patIDsMissinginDLTest]))
        datasetsUnpairedA=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),
                                       modalities=modalities, batch_size=16,exclude_patients=excludeForUnpairedA)
        dataUnpairedTestA=datasetsUnpairedA['test']
        if len(dataUnpairedTestA)>0:
            unpairedResultsA = evaluation.Evaluation(model=multisurv, dataset=dataUnpairedTestA, device=device)
            unpairedResultsA.run_bootstrap()
            create_appendline("unpairedA",unpairedResultsA,weightDir)
        
        
        excludeForUnpairedB=list(itertools.chain(*[patIDsPaired,patIDsUnpairedA,patIDsMissinginDLTest]))
        datasetsUnpairedB=utils.get_dataloaders(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'),
                                       modalities=modalities, batch_size=16,exclude_patients=excludeForUnpairedB)
        dataUnpairedTestB=datasetsUnpairedB['test']
        if len(dataUnpairedTestB)>0:
            unpairedResultsB = evaluation.Evaluation(model=multisurv, dataset=dataUnpairedTestB, device=device)
            unpairedResultsB.run_bootstrap()
            create_appendline("unpairedB",unpairedResultsB,weightDir)

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
        
        
if __name__ == '__main__':
    if os.getcwd() not in sys.path:
        sys.path.append(os.path.join(os.getcwd()))
    args = setup_args()
    evaluateByCancerType(args.dataLocation,args.weightDir,lungOnly=args.lungOnly,fusion=args.fusion,dataLoaderLocPaired=args.dataLoaderLocPaired,dataLoaderLocUnpairedA=args.dataLoaderLocUnpairedA,dataLoaderLocUnpairedB=args.dataLoaderLocUnpairedB,gpu=args.gpu)
    
    