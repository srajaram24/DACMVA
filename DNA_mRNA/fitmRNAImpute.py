#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:31:02 2023

@author: sara
"""
import torch
import torch.utils.data

import sys

import os
sys.path.append(os.getcwd())
#print(sys.path)

import argparse

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from DNA_mRNA.TrainmRNAImpute import TrainmRNAImputeClass
from functools import partial


torch.manual_seed(42)

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=401, type=int)
    options.add_argument('-dataloaderDir', '--dataloaderDir', action="store", dest="dataloaderDir",default=None)
    options.add_argument('-mp', '--missingPercent', action="store", dest="missingPercent", default=0.0, type=float)

    options.add_argument('-gpu', '--gpu', action="store", dest="gpu",default=-1,type=int)
    options.add_argument('-dataLocationTest', '--dataLocationTest', action="store", dest="dataLocationTest",default=None)
    options.add_argument('-fixedConfig', '--fixedConfig', action="store_true", dest="fixedConfig",default=False)
    options.add_argument('-modelLocation', '--modelLocation', action="store", dest="modelLocation",default=None)

    


    return options.parse_args()

def trial_str_creator(trial):
    config=trial.config
    return "bs_{}_gam_{}_lrMS_{}_t_{}".format(config["batch_size"],config["gamma"],config["lrMS"],config["t"])

def runTrainMultiSurv(config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,modelLocation,fixedConfig=False):
    MultisurvTrainer=TrainmRNAImputeClass(config,save_dir,max_epochs,missingPercent,dataloaderDir,gpu,dataLocationTest,modelLocation,fixedConfig)    
    MultisurvTrainer.trainCM()

def fitDNAmRNA(save_dir,max_epochs,fixedConfig,missingPercent,dataloaderDir=None,gpu=False,dataLocationTest=None,modelLocation=None):
    
    #adjust config, scheudler, adjust the metric/mode, change the metric in the training class in tune report
    
    
    if fixedConfig:
        
        thisConfig={
            "batch_size" : 8,
            "latent_dims" : 32,
            "gamma" : 0.1,
            "lrMS" : 1e-5,
            "t" : 0.001,
            "cancer_type":["BLCA","BRCA","HNSC","KIRC","LGG","LIHC","LUAD","LUSC","OV","STAD"],
            "cancLabel":"All",
            "RNAValSize":55, #32 for mp0 or big mp, 25 for mp95, 55 for mp90
            "ImputeBS":32,
            "UnpairedDNA":False,
            "VAE":True,
            "tag":""
            }
        thisConfig["VAE"]=False
        
        thisConfig["t"]=0.1
        thisConfig["ImputeBS"]=32
        thisConfig["gamma"]=0.6
        thisConfig["batch_size"]=8
        runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
                          missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation,fixedConfig=True)

        
        # for bs in [4]:#[8,16,24,32]:
        #     for t in [0.1]:#[0.1,0.01,0.001]:
        #         for bsImp in [8,16,24,32,64]:
        #                 for gamma in [0.1,0.2,.3,0.4,0.5,0.6,0.7,0.8,0.9]:
              
        #                     thisConfig["t"]=t
        #                     thisConfig["ImputeBS"]=bsImp
        #                     thisConfig["gamma"]=gamma
        #                     thisConfig["batch_size"]=bs
        #                     runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
        #                                       missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation,fixedConfig=True)

        # for t in [0.01,0.001,.1]:
        #     for bsImp in [4,8,16,32,64]:
        #             for gamma in [0.1,0.2,0.4,0.6,0.8]:
        #                 thisConfig["t"]=t
        #                 thisConfig["ImputeBS"]=bsImp
        #                 thisConfig["gamma"]=gamma
        #                 runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
        #                                   missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation,fixedConfig=True)

       
        
       #0.8,0.1,0.3,0.5,0.7
    
        # tVals=[0.002]
        # gammaVals=[0.5,0.1,0.15,0.2,0.3]
        # thisConfig["ImputeBS"]=32
        # thisConfig["batch_size"]=32
        # for unp in [True]:
        #     for t in tVals:
        #         for gamm in gammaVals:
        #                 thisConfig["t"]=t
        #                 thisConfig["gamma"]=gamm
        #                 thisConfig["UnpairedDNA"]=unp
        #                 runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
        #                                   missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation,fixedConfig=True)
                
        # t=0.001
        # gamm=0.2
        # thisConfig["t"]=t
        # thisConfig["gamma"]=gamm
        # runTrainMultiSurv(config=thisConfig,save_dir=save_dir, max_epochs=max_epochs, 
        #                   missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation,fixedConfig=True)
        
        
    else:
        metric='concordValA'
        mode='max'
        
        n_cpus=os.cpu_count()
        if gpu>-1:
            n_gpus=1
        else:
            n_gpus=0
            
        ray.init(num_cpus=n_cpus, num_gpus=n_gpus, local_mode=False)
        
        
        tune_config={
            "lrAE": tune.choice([0.0001,0.0005,0.0009,5e-5]),
            "lrD": tune.choice([5e-5,5e-4,1e-4]),
            "beta": tune.choice([1.0,2.0]),
            "batch_size" : [32,64,128],
            "latent_dims" : 512,
            "alpha" : tune.choice([0.1,0.5,1.0]),
            }
        
 
    
        scheduler = ASHAScheduler(
                metric=metric,
                mode=mode,
                max_t=args.max_epochs,
                grace_period=40, 
                reduction_factor=4)
        
      
        
        local_dir=save_dir
        result = tune.run(
            partial(runTrainMultiSurv, save_dir=save_dir, max_epochs=max_epochs, 
                    missingPercent=missingPercent, dataloaderDir=dataloaderDir, gpu=gpu,dataLocationTest=dataLocationTest,modelLocation=modelLocation),
            resources_per_trial={"cpu": 2, "gpu": n_gpus},
            config=tune_config,
            num_samples=200,
            scheduler=scheduler,
            checkpoint_at_end=True,
            local_dir=local_dir,
            trial_name_creator=trial_str_creator
            )
        checkpoint_path = result.get_best_checkpoint(result.get_best_trial(metric, mode), metric, mode)
        print('Best checkpoint found:', checkpoint_path)
        ray.shutdown()
    
    

if __name__ == '__main__':
    args = setup_args()
    if not torch.cuda.is_available():
        args.gpu = -1
    fitDNAmRNA(args.save_dir,args.max_epochs,fixedConfig=args.fixedConfig,
         missingPercent=args.missingPercent,dataloaderDir=args.dataloaderDir,gpu=args.gpu,dataLocationTest=args.dataLocationTest,modelLocation=args.modelLocation)
    




