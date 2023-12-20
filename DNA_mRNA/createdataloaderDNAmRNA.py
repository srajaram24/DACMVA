#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 18:26:26 2023

@author: sara
"""
import os,sys
os.chdir('/Users/sara/Desktop/modalimputation')
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))
from multisurv.src import utilsMod2

import pickle
import random

import shutil

def delete_rand_items(items,n):
    to_delete = set(random.sample(range(len(items)),n))
    return [x for i,x in enumerate(items) if not i in to_delete]

batchSize=32
missingPercent=0.95

dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModels'

mRNAPatIDs=os.listdir(os.path.join(dataLocation,"RNA-seq"))
DNAPatIDs=os.listdir(os.path.join(dataLocation,"DNAm/5k"))

if missingPercent>0:
    numToDelete=len(DNAPatIDs)-len(mRNAPatIDs)-missingPercent*len(DNAPatIDs)
    numToDelete=int(-1*numToDelete)
    mRNAPatIDs=delete_rand_items(mRNAPatIDs,numToDelete)
    dataLocationNew='/Users/sara/Desktop/multisurvAll/data/dataForModelsmp'+str(int(missingPercent*100))
    os.makedirs(os.path.join(dataLocationNew,'RNA-seq'), exist_ok=False) 
    for patID in mRNAPatIDs:
        shutil.copy(os.path.join(dataLocation,'RNA-seq',patID), os.path.join(dataLocationNew,'RNA-seq',patID)) 
    shutil.copytree(os.path.join(dataLocation,'DNAm'),os.path.join(dataLocationNew,'DNAm'),dirs_exist_ok = False)
    shutil.copytree(os.path.join(dataLocation,'Clinical'),os.path.join(dataLocationNew,'Clinical'),dirs_exist_ok = False)
    shutil.copyfile(os.path.join(dataLocation,'labels.tsv'),os.path.join(dataLocationNew,'labels.tsv'))
    dataLocation=dataLocationNew
        
    
dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModelsmp'+str(int(missingPercent*100))

#dataloaderPairedTrain,dataloaderUnpairedmRNATrain,dataloaderUnpairedDNATrain,dataloaderPairedVal,dataloaderUnpairedmRNAVal,dataloaderUnpairedDNAVal,dataloaderPairedTest,dataloaderUnpairedmRNATest,dataloaderUnpairedDNATest=utilsMod2.get_dataloaders2(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'), modalities=["mRNA","DNAm"],return_patient_id=True,batch_size=batchSize)

dataloaderPairedTrain,dataloaderPairedVal,dataloaderPairedTest=utilsMod2.get_dataloadersPaired(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'), modalities=["mRNA","DNAm"],return_patient_id=True,batch_size=batchSize)
dataLoaderLoc='/Users/sara/Desktop/dataloadersmRNADNA'

# fileName="dataloaderPairedTest"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderPairedTest,file)

fileName="dataloaderPairedTrain"+str(batchSize)+"mp"+str(int(missingPercent*100))
with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
    pickle.dump(dataloaderPairedTrain,file)
    
fileName="dataloaderPairedVal"+str(batchSize)+"mp"+str(int(missingPercent*100))
with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
    pickle.dump(dataloaderPairedVal,file)
    
# fileName="dataloaderUnpairedmRNATest"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderUnpairedmRNATest,file)

# fileName="dataloaderUnpairedmRNATrain"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderUnpairedmRNATrain,file)

# fileName="dataloaderUnpairedmRNAVal"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderUnpairedmRNAVal,file)
    

# fileName="dataloaderUnpairedDNATest"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderUnpairedDNATest,file)

# fileName="dataloaderUnpairedDNATrain"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderUnpairedDNATrain,file)

# fileName="dataloaderUnpairedDNAVal"+str(batchSize)+"mp"+str(int(missingPercent*100))
# with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
#     pickle.dump(dataloaderUnpairedDNAVal,file)
