#!/bin/bash

#SBATCH -J resampling_meshes
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/batch/back_to_gifti_%j.out
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/batch/back_to_gifti_%j.err
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem-per-cpu 6G
#SBATCH --cpus-per-task 2
#SBATCH -t 0-02:00:00

## given the content/position HCPYA information as .shape.gii files, we resample them across ico6
# and store them in-house for processing

#cd /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/utils


## user specified
recon_dir="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/utils/no_per_vertex_demeaning"
model_path="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/SiT/2026-11-11-23:35:24-epoch_100-LR_0.005-dims_192-optim_AdamW"



## should be constant
base_dir="/ceph/chpc/rcif_datasets/hcpya/S1200"
icos_dir="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/surfaces"

ico_res=6
from_sphere_L="${icos_dir}/ico-${ico_res}.L.surf.gii"
#ico_R="${icos_dir}/ico-${ico_res}.R.surf.gii"

python3 ./recon_save_as_gifti.py --out_dir "${recon_dir}" --model_dir "${model_path}"


ico_store="${recon_dir}/original_icos"
if ! test -e "${ico_store}"; then
        mkdir "${ico_store}"
    fi


## root to pull shape data from
module load workbench

for file in "${recon_dir}"/*; do
    
    full_sphere=$(basename "$file")
    type="corrThickness"

    subject_no_id="${full_sphere%%-*}"

    id="${full_sphere#*subj-}"
    id="${id%%-*}"

    if [[ "${id}" =~ ^[0-9]+$ ]]; then
        echo "Valid positive integer"
        to_sphere_L="$base_dir/${id}/MNINonLinear/fsaverage_LR32k/${id}.L.sphere.32k_fs_LR.surf.gii"

        ## set the structure of the generated icosphere
        in_metric="${recon_dir}/${full_sphere}"
        wb_command -set-structure "${in_metric}" CORTEX_LEFT

        out_metric="${recon_dir}/${subject_no_id}_${id}_${type}_32k_fs.L.shape.gii"

        # resample from the icosphere back in to hcpya sphere
        # which you can then just display on brain surf
        wb_command -metric-resample "${in_metric}" "${from_sphere_L}" "${to_sphere_L}" BARYCENTRIC "${out_metric}"
        
        # bookkeeping
        mv "${in_metric}" "${ico_store}"



        ## want a more inflated surface
        ## TODO #####################################################################################

        cp "$base_dir/${id}/MNINonLinear/fsaverage_LR32k/${id}.L.sphere.32k_fs_LR.surf.gii" "${recon_dir}"
        cp "$base_dir/${id}/MNINonLinear/fsaverage_LR32k/${id}.L.midthickness.32k_fs_LR.surf.gii" "${recon_dir}"
    fi

done

