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

from models.sit import SiT

from torch.utils.tensorboard import SummaryWriter




def get_dataloaders(data_path, bs, bs_val):
    #loading already processed and patched cortical surfaces. 

    train_data = np.load(os.path.join(data_path,'train_data.npy'))
    train_label = np.load(os.path.join(data_path,'train_labels.npy'))

    print('training data: {}'.format(train_data.shape))

    val_data = np.load(os.path.join(data_path,'validation_data.npy'))
    val_label = np.load(os.path.join(data_path,'validation_labels.npy'))

    print('validation data: {}'.format(val_data.shape))

    train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(),
                                                    torch.from_numpy(train_label).float())

    train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                    batch_size = bs,
                                                    shuffle=True,
                                                    num_workers=16)
    
    val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(),
                                                    torch.from_numpy(val_label).float())

 
    val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)
    
    test_data = np.load(os.path.join(data_path,'test_data.npy'))
    test_label = np.load(os.path.join(data_path,'test_labels.npy')).reshape(-1)

    print('testing data: {}'.format(test_data.shape))
    print('')

    test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                    torch.from_numpy(test_label).float())

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
    testing_task = config['data']['testing_task']
    if testing_task == 'sex':
        classification_task = True
    else: 
        classification_task = False


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
                        heads=config['transformer']['heads'],
                        mlp_dim=config['transformer']['mlp_dim'],
                        pool=config['transformer']['pool'], 
                        num_patches=num_patches,
                        num_classes=config['transformer']['num_classes'],
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

    

    if classification_task:  
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        if loss == 'mse':
            criterion = nn.MSELoss(reduction='mean')
        elif loss == 'l1':
            criterion = nn.L1Loss(reduction='mean')


    print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    print('Using {} criterion'.format(criterion))

    ##############################
    ######     TRAINING     ######
    ##############################

    print('')
    print('#'*30)
    print('Starting training')
    print('#'*30)
    print('')


    best_mae = 100000000
    mae_val_epoch = 100000000
    running_val_loss = 100000000

    for epoch in range(epochs):

        running_loss = 0

        model.train()

        targets_ =  []
        preds_ = []

        for i, data in enumerate(train_loader):

            inputs, targets = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs).view(-1)

            loss = criterion(outputs, targets.float().view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            targets_.append(targets.view(-1).cpu().numpy())
            preds_.append(outputs.detach().cpu().numpy())


            writer.add_scalar('loss/train', loss.item(), epoch+1)
        
        mae_epoch = np.mean(np.abs(np.concatenate(targets_) - np.concatenate(preds_)))

        writer.add_scalar('mae/train',mae_epoch, epoch+1)
        
        if (epoch+1)%5==0:
            print('| Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(epoch+1, running_loss/(i+1), round(mae_epoch,4), optimizer.param_groups[0]['lr']))

        ##############################
        ######    VALIDATION    ######
        ##############################

        if (epoch+1)%val_epoch==0:

            running_val_loss = 0

            model.eval()

            with torch.no_grad():

                targets_ = []
                preds_ = []

                for i, data in enumerate(val_loader):

                    inputs, targets = data[0].to(device), data[1].to(device)

                    outputs = model(inputs).view(-1)
                    #import pdb;pdb.set_trace()

                    loss = criterion(outputs, targets.float().view(-1))

                    running_val_loss += loss.item()

                    targets_.append(targets.view(-1).cpu().numpy())
                    preds_.append(outputs.detach().cpu().numpy())

            writer.add_scalar('loss/val', running_val_loss, epoch+1)

            mae_val_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))

            writer.add_scalar('mae/val',mae_val_epoch, epoch+1)

            print('| Validation | Epoch - {} | Loss - {:.4f} | MAE - {:.4f} |'.format(epoch+1, running_val_loss, mae_val_epoch ))

            if mae_val_epoch < best_mae:
                best_mae = mae_val_epoch
                best_epoch = epoch+1

                df = pd.DataFrame()
                df['preds'] = np.concatenate(preds_)
                df['targets'] = np.concatenate(targets_)
                df.to_csv(os.path.join(folder_to_save_model, 'preds_test.csv'))

                config['logging']['folder_model_saved'] = folder_to_save_model
                config['results'] = {}
                config['results']['best_mae'] = float(best_mae)
                config['results']['best_epoch'] = best_epoch
                config['results']['training_finished'] = False 

                with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                    yaml.dump(config, yaml_file)

                if config['training']['save_ckpt']:
                    print('saving model checkpoint...')
                    torch.save(model.state_dict(), os.path.join(folder_to_save_model,'checkpoint.pth'))

    
    print('Final results: best model obtained at epoch {} - mean absolute error {}'.format(best_epoch,best_mae))
    torch.save(model.state_dict(), os.path.join(folder_to_save_model,'model_final.pth'))

    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results'] = {}
    config['results']['best_mae'] = float(best_mae)
    config['results']['best_epoch'] = best_epoch
    config['results']['training_finished'] = True 
        
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
                        heads=config['transformer']['heads'],
                        mlp_dim=config['transformer']['mlp_dim'],
                        pool=config['transformer']['pool'], 
                        num_patches=num_patches,
                        num_classes=config['transformer']['num_classes'],
                        num_channels=config['transformer']['num_channels'],
                        num_vertices=num_vertices,
                        dim_head=config['transformer']['dim_head'],
                        dropout=config['transformer']['dropout'],
                        emb_dropout=config['transformer']['emb_dropout'])
            
        print('**** loading checkpoint ****')
        test_model.load_state_dict(torch.load(os.path.join(folder_to_save_model,'checkpoint.pth')))

        test_model.to(device)

        test_model.eval()

        with torch.no_grad():

            targets_ = []
            preds_ = []

            for i, data in enumerate(test_loader):

                inputs, targets = data[0].to(device), data[1].to(device)

                outputs = test_model(inputs).view(-1)

                targets_.append(targets.view(-1).cpu().numpy())
                preds_.append(outputs.detach().cpu().numpy())


            
            mae_test_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))

            print('| TESTING RESULTS | MAE - {:.4f} |'.format( mae_test_epoch ))

            config['results']['testing'] = float(mae_test_epoch)

            with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                yaml.dump(config, yaml_file)

            df = pd.DataFrame()
            df['preds'] = np.concatenate(preds_)
            df['targets'] = np.concatenate(targets_)
            df.to_csv(os.path.join(folder_to_save_model, 'preds_test.csv'))

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


    config_file = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/config/SiT/training/hparams_sit.yml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
          
    # Call training
    train(config)
