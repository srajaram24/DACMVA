#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:49:28 2023

@author: sara
"""

import os,sys
import pickle

dataloaderDir="/nethome/srajaram8/dataloadersmRNADNA/mp0dl"
with open(os.path.join(dataloaderDir,"dataloaderPairedVal"+str(32)+"mp"+str(0)), 'rb') as file:
    dataloaderPairedVal = pickle.load(file)
    
with open(os.path.join(dataloaderDir,"dataloaderPairedTrain"+str(32)+"mp"+str(0)), 'rb') as file:
    dataloaderPairedTrain = pickle.load(file)
    
print("VAL LENGTH")
print(len(dataloaderPairedVal.dataset))
print("Train LENGTH")
print(len(dataloaderPairedTrain.dataset))