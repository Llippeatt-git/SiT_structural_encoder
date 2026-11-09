import os
import argparse
import yaml
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


def demean(train, val, test, eps=1e-8):

    def subject_normalize(x, eps=1e-8):
        ## take the average value, std per subject, and normalize (work with intra-subject vertices)
        mean = x.mean(axis=(1,2,3), keepdims=True)
        std = x.std(axis=(1,2,3), keepdims=True)
        return (x - mean) / (std + eps)

    # per-subject normalization
    train = subject_normalize(train, eps)
    val = subject_normalize(val, eps)
    test = subject_normalize(test, eps)

    ## take the average of each vertex across subjects (work with inter-subject vertices)
    # [B, C, N, V] --> [1, C, N, V]
    vertex_wise_map = train.mean(axis = 0, keepdims=True)
    print(f"vertex mean map dims: {vertex_wise_map.shape}")
    
    # demeaning here
    train = (train - vertex_wise_map)
    val = (val - vertex_wise_map)
    test = (test - vertex_wise_map)
    return train, val, test


def get_dataloaders(data_path, config):
    # loading and normalizing already processed and patched cortical surfaces. 

    bs = config['training']['bs']
    bs_val = config['training']['bs_val']


    train_data = np.load(os.path.join(data_path,'train_data.npy'))
    #train_label = np.load(os.path.join(data_path,'train_labels.npy'))

    val_data = np.load(os.path.join(data_path,'validation_data.npy'))
    #val_label = np.load(os.path.join(data_path,'validation_labels.npy'))

    test_data = np.load(os.path.join(data_path,'test_data.npy'))
    #test_label = np.load(os.path.join(data_path,'test_labels.npy')).reshape(-1)
    
    train_data, val_data, test_data = demean(train_data, val_data, test_data)

    ## for overfit testing
    #train_data = train_data[:32]
    #train_label = train_label[:64]
    #val_data = train_data
    #val_label = train_label
    #test_data = train_data
    #test_data = train_label

    print('training data: {}'.format(train_data.shape))
    print('validation data: {}'.format(val_data.shape))
    print('testing data: {}'.format(test_data.shape))
    print('')

    train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())

    train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                    batch_size = bs,
                                                    shuffle=True,
                                                    num_workers=16)
    
    val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float())

 
    val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)

    test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float())

    test_loader = torch.utils.data.DataLoader(test_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)
    
    return train_loader, val_loader, test_loader





## per-subject pearson correlation
def calculate_correlation(target, predicted):
    t = np.concatenate(target)
    p = np.concatenate(predicted)
    n_subjects = t.shape[0]

    t = t.reshape(n_subjects, -1)   # already demeaned
    p = p.reshape(n_subjects, -1)     # demeaned the same way

    
    r = np.corrcoef(t, p)
    return r
