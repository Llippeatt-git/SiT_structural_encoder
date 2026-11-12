import os
import argparse
import yaml
import sys
from datetime import datetime

import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

##############################
### PRE-TRAINING #############
##############################

def demean(train, val, test, eps=1e-8):

    def subject_normalize(x, eps=1e-8):
        ## normalize by mean per subject
        # [B, C, N, V] --> [B, 1, 1, 1] 
        mean = x.mean(axis=(1,2,3), keepdims=True)
        std = x.std(axis=(1,2,3), keepdims=True)
        return (x - mean) / (std + eps)

    # per-subject normalization
    train = subject_normalize(train, eps)
    val = subject_normalize(val, eps)
    test = subject_normalize(test, eps)

    ## take the average of each vertex across subjects (work with inter-subject vertices)
    # [B, C, N, V] --> [1, C, N, V]
    vertexwise_mean_map = train.mean(axis = 0, keepdims=True)
    #vertexwise_std_map = train.std(axis = 0, keepdims=True)
    print(f"vertex mean map dims: {vertexwise_mean_map.shape}")
    
    # demeaning here
    train = (train - vertexwise_mean_map)
    val = (val - vertexwise_mean_map)
    test = (test - vertexwise_mean_map)
    return train, val, test


def get_dataloaders(data_path, config):
    # loading and normalizing already processed and patched cortical surfaces. 

    bs = config['training']['bs']
    bs_val = config['training']['bs_val']


    train_data = np.load(os.path.join(data_path,'train_L_data.npy'))
    train_label = np.load(os.path.join(data_path,'train_L_labels.npy'))
    train_subjects = np.load(os.path.join(data_path, 'train_L_subjects.npy'))

    val_data = np.load(os.path.join(data_path,'validation_L_data.npy'))
    val_label = np.load(os.path.join(data_path,'validation_L_labels.npy'))
    val_subjects = np.load(os.path.join(data_path, 'validation_L_subjects.npy'))

    test_data = np.load(os.path.join(data_path,'test_L_data.npy'))
    test_label = np.load(os.path.join(data_path,'test_L_labels.npy'))
    test_subjects = np.load(os.path.join(data_path, 'test_L_subjects.npy'))
    
    #train_data, val_data, test_data = demean(train_data, val_data, test_data)

    print("train data sample : ", train_data[0][0:10])

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

    train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(),
                                                        torch.from_numpy(train_label).float(),
                                                        torch.from_numpy(train_subjects).float())

    train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                    batch_size = bs,
                                                    shuffle=True,
                                                    num_workers=16)
    
    val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(),
                                                      torch.from_numpy(val_label).float(),
                                                      torch.from_numpy(val_subjects).float())

 
    val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)

    test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                       torch.from_numpy(test_label).float(),
                                                       torch.from_numpy(test_subjects).float())

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

    
    r = stats.pearsonr(t, p, axis = 1).statistic

    return r


##############################
### POST-TRAINING ############
##############################


def save_gifti(data, filename):
    gifti_file = nib.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data))
    nib.save(gifti_file,filename)


def save_reconstruction_mae_test(save_folder, recons,
                                inputs,
                                deltas, 
                                config,
                                modifier=None
                                ):

    def load_and_save_L(sphere, sphere_title, data):
        for i in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(i)].values
            sphere[indices_to_extract,:] = data[0,i,:,:].transpose()

        save_gifti(sphere, os.path.join(folder_to_save_model,f"{sphere_title}.L.shape.gii"))



    folder_to_save_model = save_folder
    ico_res = config['mesh_resolution']['ico_res']
    ico_down_res = config['mesh_resolution']['ico_down_res']
    num_channels = config['transformer']['num_channels']
    num_patches = config[f'sub_ico_{ico_down_res}']['num_patches']
    num_vertices = config[f'sub_ico_{ico_down_res}']['num_vertices']
    
    patch_extraction_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/patch_extraction"
    indices_mesh_triangles=pd.read_csv(f'{patch_extraction_path}/triangle_indices_ico_{ico_res}_sub_ico_{ico_down_res}.csv')

    ## return subjects to dims [1, C, N, V] for interoperability
    inputs = inputs[np.newaxis, :, :, :]
    recons = recons[np.newaxis, :, :, :]
    deltas = deltas[np.newaxis, :, :, :]


    original_sphere = np.zeros((40962,num_channels),dtype=np.float32)
    new_inputs = np.transpose(inputs,(0,2,1,3))
    #save_gifti(original_sphere, os.path.join(folder_to_save_model,f"original_{file_metadata}.R.shape.gii"))

    reconstructed_sphere = np.zeros((40962,num_channels),dtype=np.float32)
    new_recons = np.transpose(recons, (0,2,1,3))
    #save_gifti(reconstructed_sphere, os.path.join(folder_to_save_model,f'recon_{file_metadata}.R.shape.gii'))

    delta_sphere = np.zeros((40962,num_channels),dtype=np.float32)
    brain_difference = np.transpose(deltas, (0,2,1,3))
    #save_gifti(delta_sphere, os.path.join(folder_to_save_model,f'delta_{file_metadata}.R.shape.gii'))

    load_and_save_L(original_sphere, f"original_{modifier}", new_inputs)
    load_and_save_L(reconstructed_sphere, f"reconstructed_{modifier}", new_recons)
    load_and_save_L(delta_sphere, f"recon_delta_{modifier}", brain_difference)


def display_single_scatter(true, pred, name):

    true = true.reshape(len(true), -1)
    pred = pred.reshape(len(pred), -1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, true, s=1, alpha=0.2, linewidths=0, rasterized=True)

    lims = [min(pred.min(), true.min()), max(pred.max(), true.max())]
    ax.plot(lims, lims, "k--", lw=0.8)   # y = x reference line
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")

    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"r = {np.corrcoef(pred, true)[0, 1]:.3f}")
    plt.tight_layout()
    plt.savefig(f"{name}", dpi=200)
    #plt.show()