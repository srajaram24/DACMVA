"""Utility methods."""


import os,sys
os.chdir('/Users/sara/Desktop/modalimputation')
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd()))

import pandas as pd
import torch

import multisurv.src.dataset as dataset
import pickle

from TrainAAEIndiv.DNAmRNAModelsExpt import MRNADecoder,DNAEncoder




def get_label_map(data_file, split_group='train'):
    """Make dictionary of patient labels.

    Parameters
    ----------
    split_group: str
        Train-val-test split group name in the survival label table to subset
        data by.
    Returns
    -------
    Dict with time and event (censor variable) values for each patient id key.
    """
    df = pd.read_csv(data_file, sep='\t')

    if split_group is not None:
        groups = list(df['group'].unique())
        assert split_group in groups, f'Accepted "split_group"s are: {groups}'

        df = df.loc[df['group'] == split_group]

    keys = list(df['submitter_id'])
    values = zip(list(df['time']), list(df['event']))

    return dict(zip(keys, values))


def getDatasetsWithImputations(dataset,netEncA,netDecB):
    
    newDataset=[]
    
    for patid in range(len(dataset.patient_ids)):
        (data,*rest)=dataset.__getitem__(patid)
        dataA=data["DNAm"]
        dataB=data["mRNA"]
        if torch.sum(dataA)!=0.0 and torch.sum(dataB)!=0.0:
            newDataset.append((data,*rest))
        else:
            if torch.sum(dataB)==0.0:
                with torch.no_grad():
                    dataA=torch.unsqueeze(dataA,0)
                    latentA,_,_=netEncA(dataA)
                    imputedB=netDecB(latentA).clone().detach()
                    imputedB=torch.squeeze(imputedB,0)
                    data["mRNA"]=imputedB
                    newDataset.append((data,*rest))
                    
   
    
    
    return newDataset




def get_dlWithImputation(data_location, labels_file, modalities, netEncA, netDecB,
                    exclude_patients=None, return_patient_id=False):
    """Instantiate PyTorch DataLoaders.

    Parameters
    ----------
    Returns
    -------
    Dict of Pytorch Dataloaders.
    """
    data_dirs = {
        'clinical': os.path.join(data_location, 'Clinical'),
        'wsi': os.path.join(data_location, 'WSI'),
        'mRNA': os.path.join(data_location, 'RNA-seq'),
        'miRNA': os.path.join(data_location, 'miRNA-seq'),
        'DNAm': os.path.join(data_location, 'DNAm/5k'),
        'CNV': os.path.join(data_location, 'CNV'),
    }

    data_dirs = {mod: data_dirs[mod] for mod in modalities}


    patient_labels = {'train': get_label_map(labels_file, 'train'),
                      'val': get_label_map(labels_file, 'val'),
                      'test': get_label_map(labels_file, 'test')}


    transforms = {'train': None, 'val': None, 'test': None}#'val': None, 'test': None}
    print("DEFINE DATASET")

    datasets = {x: dataset.MultimodalDataset(
        label_map=patient_labels[x],
        data_dirs=data_dirs,
        n_patches=None,
        patch_size=None,
        transform=transforms[x],
        exclude_patients=exclude_patients,
        return_patient_id=return_patient_id,
        numCategoricalClinical=9)
                for x in ['train', 'val', 'test']} #for x in ['train', 'val', 'test']
    

    print("START ITERATING THROUGH TRAIN SET")
    
    datasetTrain=getDatasetsWithImputations(datasets["train"],netEncA,netDecB)
            
    print("FINISH ITERATING THROUGH TRAIN SET")
    
    print("START ITERATING THROUGH VAL SET")
    
    # datasetVal=getDatasetsWithImputations(datasets["val"],netEncA,netDecB)
    # print("FINISH ITERATING THROUGH VAL SET")
            
   
        
        
    
    print('Data modalities:')
    for mod in modalities:
        print('  ', mod)
    print()
    print('Dataset sizes (# patients):')
    for x in datasets.keys():
        print(f'   {x}: {len(datasets[x])}')
    print()
    #print('Batch size:', batch_size)
    print('Number of samples in training set:', len(datasetTrain))
    # print('Number of samples in val set:', len(datasetVal))
    
    

    allDL=dict()
    for b in [4,8,16,24,32,64]:
        dataloaderTrain=torch.utils.data.DataLoader(datasetTrain, batch_size=b, shuffle=True, drop_last=True)
        allDL[b]=dataloaderTrain
    
    
    
    
   
    
    
  

    return allDL



missingPercent=0.95
modelLocation="/Users/sara/Desktop/AAEAdvCox/mp"+str(int(missingPercent*100))
dataLoaderLoc='/Users/sara/Desktop/dataloadersmRNADNA/coxmp'+str(int(missingPercent*100))+'dlImputedAEAdv'
VAE=True

os.makedirs(dataLoaderLoc, exist_ok=True)
latentDim=32
cancerTypeLabel="All"
cancerTypeSubset=["BLCA","BRCA","HNSC","KIRC","LGG","LIHC","LUAD","LUSC","OV","STAD"]#['ACC','KICH','KIRC','KIRP','LGG','LIHC','LUAD','MESO','PAAD','PRAD','SKCM','UCEC','THCA']
if missingPercent==0.0:
    dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModels'
else:
    dataLocation='/Users/sara/Desktop/multisurvAll/data/dataForModelsmp'+str(int(missingPercent*100))
    
if cancerTypeSubset[0]=="All":
    exclude_patients = []
else:
    cancer_types=cancerTypeSubset
    labels = pd.read_csv(os.path.join(dataLocation,'labels.tsv'), sep='\t')
    exclude_patients = list(labels.loc[~labels['project_id'].isin(cancer_types), 'submitter_id'])

netEncA = DNAEncoder(latentdim=latentDim,VAE=VAE) 
netDecB = MRNADecoder(latentdim=latentDim)  
netEncA.load_state_dict(torch.load(os.path.join(modelLocation,"netEncA.pth")))
netDecB.load_state_dict(torch.load(os.path.join(modelLocation,"netDecB.pth")))


allDL=get_dlWithImputation(data_location=dataLocation, labels_file=os.path.join(dataLocation,'labels.tsv'), modalities=["mRNA","DNAm"],netEncA=netEncA,netDecB=netDecB,exclude_patients=exclude_patients,return_patient_id=True)



for key in allDL.keys():
    fileName="dataloader"+cancerTypeLabel+"ImputedTrain"+str(key)+"mp"+str(int(missingPercent*100))
    with open(os.path.join(dataLoaderLoc,fileName),'wb') as file:
        pickle.dump(allDL[key],file)
    

