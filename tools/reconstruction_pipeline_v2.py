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
import time
#import timm
from datetime import datetime

sys.path.append('./')
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from models.sit_recon_encoder import SiT

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr

from utils.utils import get_dataloaders, calculate_correlation
from utils.experiment_log import (log_experiment, hparam_summary,
                                  raise_on_sigterm, status_from_exception)


def individual_correlation(targets, preds, group_mean):
    '''
    Per-subject Pearson r after subtracting the train vertex-wise mean from BOTH
    target and prediction. Unlike the raw correlation, this is ~0 for a model that
    only outputs the group mean, so it measures whether individual differences
    (what the latent needs to carry for cross-modal translation) are reconstructed.
    
    ~ got from claude
    
    '''
    t = [x - group_mean for x in targets]
    p = [x - group_mean for x in preds]
    return np.nanmean(calculate_correlation(t, p))


def train(config, run_record=None):
    '''
    run_record: dict filled in as training progresses (run dir, metrics). The caller
    appends it to experiment_log.jsonl, so partial results survive a crash or timeout.
    '''
    if run_record is None:
        run_record = {}
    metrics = run_record.setdefault('metrics', {})

    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #mesh_resolution
    ico_res = config['mesh_resolution']['ico_res']
    ico_down_res = config['mesh_resolution']['ico_down_res']
    num_patches = config['sub_ico_{}'.format(ico_down_res)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(ico_down_res)]['num_vertices']

    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    loss = config['training']['loss']
    epochs = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']
    dim = config['transformer']['dim']

    folder_to_save_model = config['logging']['folder_to_save_model']
    
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print('')
    print('#'*30)
    print('Config')
    print('#'*30)
    print('')

    print('gpu: {}'.format(device))
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


    data_path = config['data']['path_to_workdir']
    train_loader, val_loader, test_loader = get_dataloaders(data_path, config)

    # Train vertex-wise mean [1, C, N, V]: the "group mean" a collapsed model would output.
    group_mean = train_loader.dataset.tensors[0].mean(dim=0, keepdim=True).numpy()
    val_inputs = val_loader.dataset.tensors[0].numpy()
    baseline_val_mae = np.mean(np.abs(val_inputs - group_mean))
    print(f'Baseline val MAE (predict train group mean for everyone): {baseline_val_mae:.6f}')
    print('A model is only learning individual structure if it beats this.')

    metrics['baseline_val_mae'] = float(baseline_val_mae)
    metrics['n_train'] = len(train_loader.dataset)
    metrics['n_val'] = len(val_loader.dataset)
    metrics['n_test'] = len(test_loader.dataset)

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

    folder_to_save_model = f"{folder_to_save_model}-epoch_{epochs}-LR_{LR}-dims_{dim}-optim_{config['optimisation']['optimiser']}"

    if not os.path.exists(folder_to_save_model):
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))


    
    run_record['run_dir'] = folder_to_save_model

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

    if config['training']['init_weights']=='recon':
        print('Loading weights from self-supervision training SiT')
        # this pipeline saves a plain state_dict; older checkpoints wrapped it in 'model_state_dict'
        state = torch.load(os.path.join(config['training']['weights_source'],'best_model.pth'), map_location='cpu')
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=True)
    else:
        print('Training from scratch')

        
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
        raise ValueError(f"optimiser '{config['optimisation']['optimiser']}' not implemented")



    if loss == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif loss == 'l1':
        criterion = nn.L1Loss(reduction='mean')
    else:
        raise ValueError(f"loss '{loss}' not implemented (use mse or l1)")



    ###############################
    #####     PRETRAINING     #####
    ###############################

    # gradient clipping (max global L2 norm); null/0 in config disables it
    grad_clip = config['training'].get('grad_clip', None)

    # LR schedule, stepped once per optimiser step (batch), not per epoch
    sched_cfg = config.get('scheduler', {}) or {}
    sched_name = sched_cfg.get('name', 'none')
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    if sched_name == 'warmup_cosine':
        warmup_steps = int(sched_cfg.get('warmup_epochs', 0) * steps_per_epoch)
        min_lr_ratio = float(sched_cfg.get('min_lr', 0.0)) / LR

        def lr_lambda(step):
            # linear warmup 0 -> LR, then cosine decay LR -> min_lr
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(progress, 1.0)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f'Using warmup + cosine schedule: {warmup_steps} warmup steps, '
              f'{total_steps} total steps, min LR {sched_cfg.get("min_lr", 0.0)}')
    elif sched_name == 'none':
        scheduler = None
        print('No LR scheduler (constant LR)')
    else:
        raise ValueError(f"scheduler '{sched_name}' not implemented (use warmup_cosine or none)")

    print(f'Gradient clipping: {grad_clip if grad_clip else "off"}')


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics['n_params'] = n_params
    print('Number of parameters: {:,}'.format(n_params))
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
    best_corr = -1
    best_corr_ind = None

    for epoch in range(epochs):

        # ---------------- TRAINING ----------------
        model.train()

        targets_ = []
        preds_ = []
        train_loss_sum = 0.0
        grad_norm_sum = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels, _ = data[0].to(device).float(), data[1].to(device).float(), data[2].to(device).float()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss_value = criterion(outputs, inputs)

            loss_value.backward()

            if grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))  # just measures
            grad_norm_sum += grad_norm.item()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += loss_value.item()

            targets_.append(inputs.cpu().numpy())
            preds_.append(outputs.cpu().detach().numpy())


        train_loss = train_loss_sum / (i + 1)
        train_mae = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))

        train_corr = np.nanmean(calculate_correlation(targets_, preds_))
        train_corr_ind = individual_correlation(targets_, preds_, group_mean)

        writer.add_scalar('loss/train', train_loss, epoch + 1)
        writer.add_scalar('mae/train', train_mae, epoch + 1)
        writer.add_scalar('corr/train', train_corr, epoch + 1)
        writer.add_scalar('corr_individual/train', train_corr_ind, epoch + 1)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('grad_norm/train', grad_norm_sum / (i + 1), epoch + 1)  # mean pre-clip norm

        metrics['epochs_completed'] = epoch + 1
        metrics['final_train_loss'] = float(train_loss)
        metrics['final_train_mae'] = float(train_mae)
        metrics['final_train_corr'] = float(train_corr)
        metrics['final_train_corr_individual'] = float(train_corr_ind)
        metrics['final_grad_norm'] = float(grad_norm_sum / (i + 1))
        metrics['final_lr'] = float(optimizer.param_groups[0]['lr'])

        # ---------------- VALIDATION ----------------
        if (epoch + 1) % val_epoch == 0 or epoch == epochs - 1:

            model.eval()


            val_targets_ = []
            val_preds_ = []
            val_loss_sum = 0.0

            with torch.no_grad():

                for i, data in enumerate(val_loader):

                    inputs, labels, _ = data[0].to(device).float(), data[1].to(device).float(), data[2].to(device).float()

                    outputs = model(inputs)
                    val_loss = criterion(outputs, inputs)



                    val_loss_sum += val_loss.item()

                    val_targets_.append(inputs.cpu().numpy())
                    val_preds_.append(outputs.cpu().detach().numpy())

            val_loss = val_loss_sum / (i + 1)
            val_mae = np.mean(np.abs(np.concatenate(val_targets_)- np.concatenate(val_preds_)))


            ## mean of subject-level correlations
            val_corr = np.nanmean(calculate_correlation(val_targets_, val_preds_))
            val_corr_ind = individual_correlation(val_targets_, val_preds_, group_mean)

            writer.add_scalar('loss/val', val_loss, epoch + 1)
            writer.add_scalar('mae/val', val_mae, epoch + 1)
            writer.add_scalar('corr/val', val_corr, epoch + 1)
            writer.add_scalar('corr_individual/val', val_corr_ind, epoch + 1)

            print(
                f'Epoch {epoch + 1}/{epochs} | '
                f'Train loss: {train_loss:.6f} | '
                f'Train MAE: {train_mae:.6f} | '
                f'Train corr: {train_corr:.6f} | '
                f'Train indiv corr: {train_corr_ind:.6f} | '
                f'Val loss: {val_loss:.6f} | '
                f'Val MAE: {val_mae:.6f} (baseline {baseline_val_mae:.6f}) | '
                f'Val corr: {val_corr:.6f} | '
                f'Val indiv corr: {val_corr_ind:.6f}'
            )

            metrics['final_val_loss'] = float(val_loss)
            metrics['final_val_mae'] = float(val_mae)
            metrics['final_val_corr'] = float(val_corr)
            metrics['final_val_corr_individual'] = float(val_corr_ind)

            # Save the best model
            if val_mae < best_mae:
                best_mae = val_mae
                best_corr = val_corr
                best_epoch = epoch + 1
                best_corr_ind = val_corr_ind

                metrics['best_epoch'] = best_epoch
                metrics['best_val_mae'] = float(best_mae)
                metrics['best_val_corr'] = float(best_corr)
                metrics['best_val_corr_individual'] = float(best_corr_ind)

                torch.save(
                    model.state_dict(),
                    os.path.join(folder_to_save_model, 'best_model.pth')
                )

    if best_epoch is None:
        raise RuntimeError("No valid best model was selected.")

    print(
        f'Best model: epoch {best_epoch}, '
        f'validation MAE {best_mae:.6f}, '
        f'validation Corr {best_corr:.6f}'
    )


    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results'] = {}
    config['results']['best_mae'] = float(best_mae)
    config['results']['best_corr'] = float(best_corr)
    config['results']['best_epoch'] = best_epoch
    config['results']['baseline_val_mae'] = float(baseline_val_mae)
    config['results']['training_finished'] = True

    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    writer.close()

    # This file intentionally represents the last epoch, not the best.
    torch.save(
        model.state_dict(),
        os.path.join(folder_to_save_model, 'model_final.pth')
    )

    ##############################
    ######     TESTING      ######
    ##############################

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
    test_model.load_state_dict(torch.load(os.path.join(folder_to_save_model,'best_model.pth'), map_location=device))

    test_model.to(device)

    test_model.eval()

    with torch.no_grad():

        test_targets_ = []
        test_preds_ = []

        for data in test_loader:

            inputs, labels, _ = data[0].to(device).float(), data[1].to(device).float(), data[2].to(device).float()

            outputs = test_model(inputs)

            test_targets_.append(inputs.cpu().numpy())
            test_preds_.append(outputs.cpu().detach().numpy())

        test_mae = np.mean(np.abs(np.concatenate(test_targets_)- np.concatenate(test_preds_)))


        ## correlation code
        test_corr = np.nanmean(calculate_correlation(test_targets_, test_preds_))
        test_corr_ind = individual_correlation(test_targets_, test_preds_, group_mean)

        print('| TESTING RESULTS | MAE - {:.4f} | Corr - {:.4f} | Indiv corr - {:.4f}'.format(test_mae, test_corr, test_corr_ind))

        config['results']['test_accuracy'] = float(test_mae)
        config['results']['test_correlation'] = float(test_corr)
        config['results']['test_individual_correlation'] = float(test_corr_ind)

        metrics['test_mae'] = float(test_mae)
        metrics['test_corr'] = float(test_corr)
        metrics['test_corr_individual'] = float(test_corr_ind)

        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SiT_structural_encoder')

    parser.add_argument('--config', type=str, default='../config/hparams_recon.yml',
                        help='path where the configs are stored')
    
    args = parser.parse_args()
    config_file = args.config
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
          
    # One record per run goes to experiment_log.jsonl, including failed/killed runs.
    run_record = {
        'run_type': 'train',
        'status': 'running',
        'config_file': os.path.abspath(config_file),
        'run_dir': None,
        'hparams': hparam_summary(config),
        'metrics': {},
    }
    raise_on_sigterm()
    start = time.time()
    try:
        train(config, run_record)
        run_record['status'] = 'completed'
    except BaseException as e:
        run_record['status'] = status_from_exception(e)
        run_record['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        run_record['duration_s'] = round(time.time() - start, 1)
        log_experiment(run_record, config.get('logging', {}).get('experiment_log'))
