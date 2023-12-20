#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:23:10 2023

@author: sara
"""
shapLabel="THCA"
fileNameTest="dataTestTHCA"


with open(os.path.join('/Users/sara/Desktop/mp60models/bestRNAImpute',"shapValues"+shapLabel),'rb') as file:
    shapValuesBest=pickle.load(file)
    
with open(os.path.join('/Users/sara/Desktop/mp60models/baseline',"shapValues"+shapLabel),'rb') as file:
    shapValuesBase=pickle.load(file)


with open(os.path.join('/Users/sara/Desktop/mp60models/',fileNameTest),'rb') as file:
    datasetTest=pickle.load(file)

(data,time,event,*rest)=datasetTest
dataTest=data["mRNA"]

    
i=0  
shap.summary_plot(shapValuesBest[i],dataTest,feature_names=genes, plot_type="bar",show=False)
plt.title(shapLabel+" Best mp60 Mode1 t="+str(i))
plt.show()

shap.summary_plot(shapValuesBase[i],dataTest,feature_names=genes, plot_type="bar",show=False)
plt.title(shapLabel+" Base mp60 Mode1 t="+str(i))
plt.show()