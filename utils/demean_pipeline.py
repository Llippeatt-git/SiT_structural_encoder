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

from models.sit_recon_encoder import SiT
from utils.utils import save_gifti

data_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA"
patch_extraction_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/patch_extraction"
indices_mesh_triangles=pd.read_csv(f'{patch_extraction_path}/triangle_indices_ico_6_sub_ico_2.csv')


def save_that_subject(input, subject_id, name, folder):
    input = input[np.newaxis, :, :, :]
    
    np.save(f"{folder}/{name}", input)

    original_sphere = np.zeros((40962,1),dtype=np.float32)
    new_inputs = np.transpose(input,(0,2,1,3))

    for i in range(320):
        indices_to_extract = indices_mesh_triangles[str(i)].values
        original_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(original_sphere, os.path.join(folder,f"{name}-{subject_id}-.L.shape.gii"))


def main(folder_to_save, idx):
    
    train_data = np.load(os.path.join(data_path,'train_L_data.npy'))
    val_data = np.load(os.path.join(data_path,'validation_L_data.npy'))
    test_data = np.load(os.path.join(data_path,'test_L_data.npy'))
    test_subjects = np.load(os.path.join(data_path, 'test_L_subjects.npy'))


    subject_id = int(test_subjects[idx][0])

    
    save_that_subject(test_data[idx], subject_id, f"pre_demeaning_subject", folder_to_save)

    def subject_normalize(x, eps=1e-8):
        ## take the average value, std per subject, and normalize (work with intra-subject vertices)
        mean = x.mean(axis=(1,2,3), keepdims=True)
        std = x.std(axis=(1,2,3), keepdims=True)
        return (x - mean) / (std + eps)

    # per-subject normalization
    train_data = subject_normalize(train_data)
    val_data = subject_normalize(val_data)
    test_data = subject_normalize(test_data)

    save_that_subject(test_data[idx], subject_id, "per_subject_normalized_subject", folder_to_save)

    ## take the average of each vertex across subjects (work with inter-subject vertices)
    # [B, C, N, V] --> [1, C, N, V]
    vertex_wise_map = train_data.mean(axis = 0, keepdims=True)
    print(f"vertex mean map dims: {vertex_wise_map.shape}")

    # demeaning here
    train_data = (train_data - vertex_wise_map)
    val_data = (val_data - vertex_wise_map)
    test_data = (test_data - vertex_wise_map)

    save_that_subject(test_data[idx], subject_id, "full_demeaned_subject", folder_to_save)
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save reconstructions as GIFTI')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='folder to write the .shape.gii files into')
    parser.add_argument('--subject_index_get', type=str, required=True,
                            help='index to pull information from')
    args = parser.parse_args()
    out_dir = args.out_dir
    idx = int(args.subject_index_get)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('Creating folder: {}'.format(out_dir))

    main(out_dir, idx)