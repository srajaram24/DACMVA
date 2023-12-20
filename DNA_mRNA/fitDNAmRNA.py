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
from DNA_mRNA.TrainDNAmRNA import TrainDNAmRNAClass
from functools import partial


torch.manual_seed(42)

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=401, type=int)
    options.add_argument('-dataloaderDir', '--dataloaderDir', action="store", dest="dataloaderDir",default=None)
    options.add_argument('-mp', '--missingPercent', action="store", dest="missingPercent", default=0.0, type=float)

    options.add_argument('-gpu', '--use-gpu', action="store_true", dest="use_gpu")
    options.add_argument('-fixedConfig', '--fixedConfig', action="store_true", dest="fixedConfig",default=False)
    options.add_argument('-modelLocation', '--modelLocation', action="store", dest="modelLocation",default=None)
    options.add_argument('-modelLocEpoch', '--modelLocEpoch', action="store", dest="modelLocEpoch",default=-1,type=int)
    


    return options.parse_args()

def trial_str_creator(trial):
    thisConfig=trial.config
    return "lrAE_{}_lrD_{}_bs_{}_beta_{}_alpha_{}".format(thisConfig["lrAE"],thisConfig["lrD"],thisConfig["batch_size"],thisConfig["beta"],thisConfig["alpha"])

def runTrainMultiSurv(config,save_dir,save_freq,max_epochs,missingPercent,dataloaderDir,use_gpu,modelLocation,modelLocEpoch,fixedConfig=False):
    MultisurvTrainer=TrainDNAmRNAClass(config,save_dir,save_freq,max_epochs,missingPercent,dataloaderDir,use_gpu,modelLocation,modelLocEpoch,fixedConfig)    
    MultisurvTrainer.trainCM()

def fitDNAmRNA(save_dir,save_freq,max_epochs,fixedConfig=False,missingPercent=0.0,dataloaderDir=None,use_gpu=False,modelLocation=None,modelLocEpoch=-1):
    
    #adjust config, scheudler, adjust the metric/mode, change the metric in the training class in tune report
    
    
    if fixedConfig:
        
        thisConfig={
            "lrAE": 0.0001,
            "lrD": 5e-5,
            "beta": 1.0,
            "batch_size" : 64,
            "latent_dims" : 512,
            "alpha" : 1.0,
            }
        
    
        runTrainMultiSurv(config=thisConfig,save_dir=save_dir, save_freq=save_freq, max_epochs=max_epochs, 
                missingPercent=missingPercent, dataloaderDir=dataloaderDir, use_gpu=use_gpu,modelLocation=modelLocation,modelLocEpoch=modelLocEpoch,fixedConfig=True)
        
    else:
        metric='concordValA'
        mode='max'
        
        n_cpus=os.cpu_count()
        if use_gpu:
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
            partial(runTrainMultiSurv, save_dir=save_dir, save_freq=save_freq, max_epochs=max_epochs, 
                    missingPercent=missingPercent, dataloaderDir=dataloaderDir, use_gpu=use_gpu,modelLocation=modelLocation,modelLocEpoch=modelLocEpoch),
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
        args.use_gpu = False
    fitDNAmRNA(args.save_dir,args.save_freq,args.max_epochs,fixedConfig=args.fixedConfig,
         missingPercent=args.missingPercent,dataloaderDir=args.dataloaderDir,use_gpu=args.use_gpu,modelLocation=args.modelLocation,modelLocEpoch=args.modelLocEpoch)
    




