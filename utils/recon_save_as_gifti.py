import pandas as pd
import nibabel as nib
import numpy as np
import yaml
import os
import sys
import torch
from datetime import datetime
import argparse

sys.path.append('./')
sys.path.append('../')

from einops.layers.torch import Rearrange
from einops import rearrange

from models.sit_recon_encoder import SiT
from utils import get_dataloaders, calculate_correlation, save_reconstruction_mae_test, display_single_scatter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''
Given a trained model, and the associated dataset it was trained on, we run the model on the test set to obtain the
target/reconstruction pairs. 
We take the pairs that have the highest, lowest, and median correlations, then we save them as giftis to be
inflated back into ico6, then brain
'''


def main(model_path, folder_to_save, config):
    data_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA"
    ico_res = 6
    ico_down_res = 2


    bulk_data_folder = f"{folder_to_save}/bulk_data"
    if not os.path.exists(bulk_data_folder):
        os.mkdir(bulk_data_folder)
        print('Creating folder: {}'.format(bulk_data_folder))

    _, _, test_loader = get_dataloaders(data_path, config)


    test_model = SiT(dim=config['transformer']['dim'],
                depth=config['transformer']['depth'],
                dec_depth=config['transformer']['dec_depth'],
                heads=config['transformer']['heads'],
                mlp_dim=config['transformer']['mlp_dim'],
                latent_dim=config['transformer']['latent_dim'],
                pool=config['transformer']['pool'], 
                num_patches=config['sub_ico_{}'.format(ico_down_res)]['num_patches'],
                num_channels=config['transformer']['num_channels'],
                num_vertices=config['sub_ico_{}'.format(ico_down_res)]['num_vertices'],
                dim_head=config['transformer']['dim_head'],
                dropout=config['transformer']['dropout'],
                emb_dropout=config['transformer']['emb_dropout'])
    
    test_model.load_state_dict(torch.load(os.path.join(model_path,'best_model.pth')))
    test_model.to(device)


    test_model.eval()

    with torch.no_grad():

        test_targets_ = []
        test_preds_ = []
        test_subjects_ = []

        for data in test_loader:

            inputs, _, subjects = data[0].to(device).float(), data[1].to(device).float(), data[2].to(device).int()

            outputs = test_model(inputs)

            test_targets_.append(inputs.cpu().numpy())
            test_preds_.append(outputs.cpu().detach().numpy())
            test_subjects_.append(subjects.cpu().numpy())


        t = np.concatenate(test_targets_)
        p = np.concatenate(test_preds_)
        s = np.concatenate(test_subjects_)

        np.save(f"{bulk_data_folder}/targets_data.npy", t)
        np.save(f"{bulk_data_folder}/predicted_data.npy", p)
        np.save(f"{bulk_data_folder}/subjects_data.npy", s)

        test_mae = np.mean(np.abs(np.concatenate(test_targets_)- np.concatenate(test_preds_)))
        print('| TESTING RESULTS | MAE - {:.4f}'.format( test_mae ))
        
        test_corr = calculate_correlation(test_targets_, test_preds_)

        df = pd.DataFrame({'subject': np.arange(len(test_corr)), 'corr': test_corr})
        df.to_csv(f'{bulk_data_folder}/subject_correlations.csv', index=False)

        # once we have the correlations dataframe, sort by correlation so we can identify highest/lowest/median
        df_sorted = df.sort_values('corr')



        ## create new folders for better organization
        specific_recons = f"{folder_to_save}/recons_snapshot"
        if not os.path.exists(specific_recons):
            os.mkdir(specific_recons)
            print('Creating folder: {}'.format(specific_recons))
        
        # metadata list of tuples, stores the indicator (for naming purposes) and the associated index
        metadata = [("high", -1), ("median", int(len(df)/2)), ("low", 0)]
        for name, i in metadata:

            idx = int(df_sorted.iloc[i].subject)
            print(f"{name} correlation {df_sorted.iloc[-1].corr}, found at id {idx}")
        
            target_brain_data = t[idx]
            recon_brain_data = p[idx]
            delta = target_brain_data - recon_brain_data

            subject = s[idx][0]
            print(subject)

            np.save(f"{specific_recons}/{name}_target_subj-{subject}-.npy", target_brain_data)
            np.save(f"{specific_recons}/{name}_recon_subj-{subject}-.npy", recon_brain_data)
            np.save(f"{specific_recons}/{name}_delta_subj-{subject}-.npy", delta)

            brain_modifier = f"{name}_correlation_subj-{subject}-"
            print(f"saving {brain_modifier}...")

            display_single_scatter(target_brain_data, recon_brain_data, f"{specific_recons}/{brain_modifier}")
            save_reconstruction_mae_test(folder_to_save, recon_brain_data, target_brain_data, delta, config, brain_modifier)

        

        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save reconstructions as GIFTI')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='folder to write the .shape.gii files into')
    
    parser.add_argument('--model_dir', type=str, required=True,
                        help='folder to pull configs and data from')
    args = parser.parse_args()
    out_dir = args.out_dir
    model_path = args.model_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('Creating folder: {}'.format(out_dir))

    config_path = model_path + "/hparams.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    main(model_path, out_dir, config)