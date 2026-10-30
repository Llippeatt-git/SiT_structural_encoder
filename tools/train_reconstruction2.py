# -*- coding: utf-8 -*-
# @Author: Simon Dahan
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#


'''
This file implements the training procedure to train a SiT/MS-SiT model.
Models can be trained for two tasks:
    - age at scan prediction
    - birth age prediction

Pretrained ImageNet models are downloaded from the Timm library. 
'''

import os
import argparse
import yaml
import sys
#import timm
from datetime import datetime

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from models.sit_recon_encoder import SiT

from torch.utils.tensorboard import SummaryWriter




def get_dataloaders(data_path, bs, bs_val):
    # loading and normalizing already processed and patched cortical surfaces. 

    train_data = np.load(os.path.join(data_path,'train_data.npy'))
    #train_label = np.load(os.path.join(data_path,'train_labels.npy'))

    mean = train_data.mean()
    std = train_data.std()
    print(f'train mean: {mean:.4f} | std: {std:.4f}\n')

    

    val_data = np.load(os.path.join(data_path,'validation_data.npy'))
    #val_label = np.load(os.path.join(data_path,'validation_labels.npy'))


    test_data = np.load(os.path.join(data_path,'test_data.npy'))
    #test_label = np.load(os.path.join(data_path,'test_labels.npy')).reshape(-1)
    

    eps = 1e-8
    train_data = (train_data - mean) / (std + eps)
    val_data   = (val_data   - mean) / (std + eps)
    test_data  = (test_data  - mean) / (std + eps)


    ## for overfit testing
    train_data = train_data[:64]
    #train_label = train_label[:64]
    val_data = train_data
    #val_label = train_label
    test_data = train_data
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


def train(config):

    #mesh_resolution
    ico_res = config['mesh_resolution']['ico_res']
    ico_down_res = config['mesh_resolution']['ico_down_res']
    num_patches = config['sub_ico_{}'.format(ico_down_res)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(ico_down_res)]['num_vertices']

    #data
    task = config['data']['task']


    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    loss = config['training']['loss']
    epochs = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    testing = config['training']['testing']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']


    folder_to_save_model = config['logging']['folder_to_save_model']
    
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print('')
    print('#'*30)
    print('Config')
    print('#'*30)
    print('')

    print('gpu: {}'.format(device))
    print('task: {}'.format(task))  
    print('model: {}'.format(config['MODEL']))

    print('Mesh resolution - ico {}'.format(ico_res))
    print('Grid resolution - ico {}'.format(ico_down_res))
    print('Number of patches - {}'.format(num_patches))
    print('Number of vertices - {}'.format(num_vertices))


    ##############################
    ######     DATASET      ######
    ##############################

    print('')
    print('#'*30)
    print('Loading data')
    print('#'*30)
    print('')

    print(f'LOADING DATA: ICO {ico_res} - sub-res ICO {ico_down_res}')


    data_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA"
    train_loader, val_loader, test_loader = get_dataloaders(data_path, bs, bs_val)

    ##############################
    ######      LOGGING     ######
    ##############################

    # creating folders for logging. 
        
    if not os.path.exists(folder_to_save_model):
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # folder time
    folder_to_save_model = os.path.join(folder_to_save_model,date)
    print(folder_to_save_model)
    if config['transformer']['dim'] == 192:
        folder_to_save_model = folder_to_save_model + '-small'
    elif config['transformer']['dim'] == 384:
        folder_to_save_model = folder_to_save_model + '-medium'
    elif config['transformer']['dim'] == 768:
        folder_to_save_model = folder_to_save_model + '-large'

    folder_to_save_model = folder_to_save_model + '-recon'
    
    if not os.path.exists(folder_to_save_model):
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))
        
    writer = SummaryWriter(log_dir=folder_to_save_model)


    ##############################
    #######     MODEL      #######
    ##############################

    print('')
    print('#'*30)
    print('Init model')
    print('#'*30)
    print('')

    if config['transformer']['model'] == 'SiT':

        model = SiT(dim=config['transformer']['dim'],
                        depth=config['transformer']['depth'],
                        dec_depth=config['transformer']['dec_depth'],
                        heads=config['transformer']['heads'],
                        mlp_dim=config['transformer']['mlp_dim'],
                        latent_dim=config['transformer']['latent_dim'],
                        pool=config['transformer']['pool'], 
                        num_patches=num_patches,
                        num_channels=config['transformer']['num_channels'],
                        num_vertices=num_vertices,
                        dim_head=config['transformer']['dim_head'],
                        dropout=config['transformer']['dropout'],
                        emb_dropout=config['transformer']['emb_dropout'])
        
    model.to(device)

    if config['optimisation']['optimiser']=='Adam':
        print('using Adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config['Adam']['weight_decay'])
    elif config['optimisation']['optimiser']=='SGD':
        print('using SGD optimiser')
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        print('using AdamW optimiser')
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')

    

    if loss == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif loss == 'l1':
        criterion = nn.L1Loss(reduction='mean')


    print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    print(f'Using {criterion} criterion')

    ##############################
    ######     TRAINING     ######
    ##############################

    print('')
    print('#'*30)
    print('Starting training')
    print('#'*30)
    print('')

    best_mae = float('inf')
    best_epoch = None

    for epoch in range(epochs):

        # ---------------- TRAINING ----------------
        model.train()

        train_abs_error = 0.0
        train_numel = 0
        train_loss_sum = 0.0

        for data in train_loader:
            inputs = data[0].to(device).float()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss_value = criterion(outputs, inputs)

            loss_value.backward()
            optimizer.step()

            n = inputs.numel()
            train_loss_sum += loss_value.item() * n
            train_abs_error += (
                outputs.detach() - inputs
            ).abs().sum().item()
            train_numel += n

        if train_numel == 0:
            raise RuntimeError("Training loader is empty.")

        train_loss = train_loss_sum / train_numel
        train_mae = train_abs_error / train_numel

        writer.add_scalar('loss/train', train_loss, epoch + 1)
        writer.add_scalar('mae/train', train_mae, epoch + 1)

        # ---------------- VALIDATION ----------------
        if (epoch + 1) % val_epoch == 0 or epoch == epochs - 1:

            model.eval()

            val_abs_error = 0.0
            val_numel = 0
            val_loss_sum = 0.0

            with torch.no_grad():

                for data in val_loader:

                    inputs = data[0].to(device).float()

                    outputs = model(inputs)
                    val_loss = criterion(outputs, inputs)



                    n = inputs.numel()
                    val_loss_sum += val_loss.item() * n
                    val_abs_error += (
                        outputs - inputs
                    ).abs().sum().item()
                    val_numel += n


            val_loss_mean = val_loss_sum / val_numel
            val_mae = val_abs_error / val_numel

            writer.add_scalar('loss/val', val_loss_mean, epoch + 1)
            writer.add_scalar('mae/val', val_mae, epoch + 1)

            print(
                f'Epoch {epoch + 1}/{epochs} | '
                f'Train loss: {train_loss:.6f} | '
                f'Train MAE: {train_mae:.6f} | '
                f'Val loss: {val_loss_mean:.6f} | '
                f'Val MAE: {val_mae:.6f}'
            )

            # Save the best model
            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    os.path.join(folder_to_save_model, 'best_model.pth')
                )

    if best_epoch is None:
        raise RuntimeError("No valid best model was selected.")

    print(
        f'Best model: epoch {best_epoch}, '
        f'validation MAE {best_mae:.6f}'
    )


    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results'] = {}
    config['results']['best_mae'] = float(best_mae)
    config['results']['best_epoch'] = best_epoch
    config['results']['training_finished'] = False 

    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    # This file intentionally represents the last epoch, not the best.
    torch.save(
        model.state_dict(),
        os.path.join(folder_to_save_model, 'model_final.pth')
    )

    ##############################
    ######     TESTING      ######
    ##############################

    if testing:
        print('')
        print('#'*30)
        print('Starting testing')
        print('#'*30)
        print('')

        del model
        torch.cuda.empty_cache()
        
        if config['transformer']['model'] == 'SiT':

            test_model = SiT(dim=config['transformer']['dim'],
                depth=config['transformer']['depth'],
                dec_depth=config['transformer']['dec_depth'],
                heads=config['transformer']['heads'],
                mlp_dim=config['transformer']['mlp_dim'],
                latent_dim=config['transformer']['latent_dim'],
                pool=config['transformer']['pool'], 
                num_patches=num_patches,
                num_channels=config['transformer']['num_channels'],
                num_vertices=num_vertices,
                dim_head=config['transformer']['dim_head'],
                dropout=config['transformer']['dropout'],
                emb_dropout=config['transformer']['emb_dropout'])
            
        print('**** loading checkpoint ****')
        test_model.load_state_dict(torch.load(os.path.join(folder_to_save_model,'best_model.pth')))

        test_model.to(device)

        test_model.eval()

        with torch.no_grad():

            test_abs_error = 0.0
            test_numel = 0

            for data in test_loader:

                inputs = data[0].to(device).float()

                outputs = test_model(inputs)

                n = inputs.numel()
                test_abs_error += (
                    outputs.detach() - inputs
                ).abs().sum().item()
                test_numel += n

            test_mae = train_abs_error / train_numel

            print('| TESTING RESULTS | MAE - {:.4f} |'.format( test_mae ))

            config['results']['testing'] = float(test_mae)

            with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                yaml.dump(config, yaml_file)

    else:
        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='ViT')

    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams_sit.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    '''


    config_file = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/config/SiT/training/hparams_recon.yml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
          
    # Call training
    train(config)
