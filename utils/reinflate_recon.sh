#!/bin/bash

#SBATCH -J resampling_meshes
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA/batch/resamping_meshes_%j.out
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA/batch/resamping_meshes_%j.err
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem-per-cpu 8G
#SBATCH --cpus-per-task 2
#SBATCH -t 0-02:00:00

## given the content/position HCPYA information as .shape.gii files, we resample them across ico6
# and store them in-house for processing


## root to pull shape data from
module load workbench

recon_dir="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/SiT/recons"
base_dir="/ceph/chpc/rcif_datasets/hcpya/S1200"
icos_dir="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/surfaces"

ico_res=6
ico_L="${icos_dir}/ico-${ico_res}.L.surf.gii"
ico_R="${icos_dir}/ico-${ico_res}.R.surf.gii"

for obj in "$recon_dir"/*; do
    subj_name=$(basename "$obj")

    full_sphere="${recon_dir}/${subj_name}"
    echo "${full_sphere}"
    
    if [[ "${full_sphere}" == *".L.shape.gii" ]]; then
        wb_command -set-structure "${full_sphere}" CORTEX_LEFT
    else 
        wb_command -set-structure "${full_sphere}" CORTEX_RIGHT
    fi

done
