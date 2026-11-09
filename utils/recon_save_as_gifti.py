import pandas as pd
import nibabel as nib
import numpy as np
import yaml
import os
import sys
import torch

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from einops.layers.torch import Rearrange
from einops import rearrange

from models.sit_recon_encoder import SiT
from SiT_structural_encoder.tools.train_reconstruction_final import get_dataloaders
from scipy.stats import pearsonr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

patch_extraction_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/patch_extraction"
data_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA"

demeaned_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/SiT/2026-11-04-11:16:15-demeaned/"
nondemeaned_path = "/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/SiT/2026-11-04-11:32:21-small-recon/"
config_file = 'hparams.yml'


def calculate_correlation(target, predicted):
    t = np.concatenate(target)
    p = np.concatenate(predicted)
    n_subjects = t.shape[0]

    t = t.reshape(n_subjects, -1)   # already demeaned
    p = p.reshape(n_subjects, -1)     # demeaned the same way

    r = pearsonr(t, p, axis=1).statistic

    #r_mean = np.tanh(np.arctanh(r).mean())   # Fisher-z average
    #print(f"shameen : {r_mean}")

    return r



def save_gifti(data, filename):
    gifti_file = nib.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data))
    nib.save(gifti_file,filename)



def save_reconstruction_mae_test(recons,
                                inputs,
                                deltas, 
                                config,
                                file_metadata
                                ):

    folder_to_save_model = config['logging']['folder_model_saved']
    ico_res = config['mesh_resolution']['ico_res']
    ico_down_res = config['mesh_resolution']['ico_down_res']
    num_channels = config['transformer']['num_channels']
    num_patches = config[f'sub_ico_{ico_down_res}']['num_patches']
    num_vertices = config[f'sub_ico_{ico_down_res}']['num_vertices']

    if not os.path.exists(folder_to_save_model):
            os.mkdir(folder_to_save_model)
            print('Creating folder: {}'.format(folder_to_save_model))
    

    
    indices_mesh_triangles=pd.read_csv(f'{patch_extraction_path}/triangle_indices_ico_{ico_res}_sub_ico_{ico_down_res}.csv')

    original_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    new_inputs = np.transpose(inputs,(0,2,1,3))

    for i in range(num_patches):
        indices_to_extract = indices_mesh_triangles[str(i)].values
        original_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(original_sphere, os.path.join(folder_to_save_model,f"original_{file_metadata}.L.shape.gii"))
    save_gifti(original_sphere, os.path.join(folder_to_save_model,f"original_{file_metadata}.R.shape.gii"))



    reconstructed_sphere = np.zeros((40962,num_channels),dtype=np.float32)
    new_recons = np.transpose(recons, (0,2,1,3))
    print(new_recons.shape)
    for i in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(i)].values
            reconstructed_sphere[indices_to_extract,:] = new_recons[0,i,:,:].transpose()

    save_gifti(reconstructed_sphere, os.path.join(folder_to_save_model,f'recon_{file_metadata}.L.shape.gii'))
    save_gifti(reconstructed_sphere, os.path.join(folder_to_save_model,f'recon_{file_metadata}.R.shape.gii'))
    


    delta_sphere = np.zeros((40962,num_channels),dtype=np.float32)
    brain_difference = np.transpose(deltas, (0,2,1,3))
    print(new_recons.shape)
    for i in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(i)].values
            delta_sphere[indices_to_extract,:] = brain_difference[0,i,:,:].transpose()

    save_gifti(delta_sphere, os.path.join(folder_to_save_model,f'delta_{file_metadata}.L.shape.gii'))
    save_gifti(delta_sphere, os.path.join(folder_to_save_model,f'delta_{file_metadata}.R.shape.gii'))






def main(config):
    ico_res = 6
    ico_down_res = 2
    num_patches = config['sub_ico_{}'.format(ico_down_res)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(ico_down_res)]['num_vertices']

    train_loader, val_loader, test_loader, train_mean = get_dataloaders(data_path, config)

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

    test_model.load_state_dict(torch.load(os.path.join(demeaned_path,'best_model.pth')))


    test_model.to(device)

    test_model.eval()

    with torch.no_grad():

        test_targets_ = []
        test_preds_ = []
        latent_reps_ = []

        for data in test_loader:

            inputs = data[0].to(device).float()

            outputs = test_model(inputs)

            encoded_outputs = test_model.encode(inputs)
            latent_reps_.append(encoded_outputs.cpu().detach().numpy())

            test_targets_.append(inputs.cpu().numpy())
            test_preds_.append(outputs.cpu().detach().numpy())


        t = np.concatenate(test_targets_)
        p = np.concatenate(test_preds_)
        e = np.concatenate(latent_reps_)

        np.save("targets_data.npy", t)
        np.save("predicted_data.npy", p)
        np.save("latent_representation", e)
        
        test_mae = np.mean(np.abs(np.concatenate(test_targets_)- np.concatenate(test_preds_)))
        
        test_corr = calculate_correlation(test_targets_, test_preds_)
        df = pd.DataFrame({'subject': np.arange(len(test_corr)), 'corr': test_corr})
        df.to_csv('subject_correlations.csv', index=False)
        
        print('| TESTING RESULTS | MAE - {:.4f}'.format( test_mae ))


        df_sorted = df.sort_values('corr')
        # highest, lowest, and median correlation scores
        subject_indexes = [0, -1, int(len(df)/2)]
        for i in subject_indexes:
            idx = int(df_sorted.iloc[i].subject)

            print("id : ", idx)
            print("corr rank : ", i)
            print("corr : ", df_sorted.iloc[i].corr)

            target_brain_data = t[idx]
            recon_brain_data = p[idx]
            np.save(f"target_{idx}.npy", target_brain_data)
            np.save(f"recon_{idx}.npy", recon_brain_data)
            brain_recon_delta = target_brain_data - recon_brain_data

            target_brain_data = target_brain_data[np.newaxis, :, :, :]
            recon_brain_data = recon_brain_data[np.newaxis, :, :, :]
            brain_recon_delta = brain_recon_delta[np.newaxis, :, :, :]

            metadata = f"{idx}_rank_{i}_{ico_down_res}_to_{ico_res}_{("DEMEANED" if config['training']['demeaned'] else "MEANED")}"
            print(metadata)
            save_reconstruction_mae_test(recon_brain_data, target_brain_data, brain_recon_delta, config, metadata)


        

        
        

if __name__ == "__main__":

    ## DEMEANED VERSION

    config_path = demeaned_path + config_file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    main(config)