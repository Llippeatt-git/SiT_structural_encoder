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


base_dir="/ceph/chpc/rcif_datasets/hcpya/S1200"
icos_dir="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/surfaces"

dest_dir="/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/HCP_YA/resampled_icos"

# to-sphere for resampling
ico_res=6
ico_L="${icos_dir}/ico-${ico_res}.L.surf.gii"
ico_R="${icos_dir}/ico-${ico_res}.R.surf.gii"

echo "${ico_L}"

if test -f "$ico_L" && test -f "$ico_R"; then
    echo "icospheres found"
else
    echo "icospheres not found, please relocate them to $icos_dir"
    exit 1
fi


## create a dummy corpus of ten subjects, with all their data in place for encoding
count=0
max=1300

for subj in "$base_dir"/*; do
    count=$((count + 1))
    if [ $count -gt $max ]; then
        break
    fi

    subj_name=$(basename "$subj")
    type="corrThickness"

    output_subject="${dest_dir}/${subj_name}"
    

    # the actual position/content information
    s1200_shape_L="$base_dir/${subj_name}/MNINonLinear/fsaverage_LR32k/${subj_name}.L.${type}.32k_fs_LR.shape.gii"
    s1200_shape_R="$base_dir/${subj_name}/MNINonLinear/fsaverage_LR32k/${subj_name}.R.${type}.32k_fs_LR.shape.gii"
    
    # from sphere
    from_sphere_L="$base_dir/${subj_name}/MNINonLinear/fsaverage_LR32k/${subj_name}.L.sphere.32k_fs_LR.surf.gii"
    from_sphere_R="$base_dir/${subj_name}/MNINonLinear/fsaverage_LR32k/${subj_name}.R.sphere.32k_fs_LR.surf.gii"
    


    if ! test -e "${output_subject}"; then
        mkdir "${output_subject}"
    fi

    # resampled pos/content info
    resamp_L="${output_subject}/${subj_name}_ico${ico_res}.L.shape.gii"
    resamp_R="${output_subject}/${subj_name}_ico${ico_res}.R.shape.gii"
    if ! test -f "$resamp_L"; then
        echo "${subj_name}"

        wb_command -metric-resample "${s1200_shape_L}" "${from_sphere_L}" "${ico_L}" BARYCENTRIC "${resamp_L}"
        wb_command -metric-resample "${s1200_shape_R}" "${from_sphere_R}" "${ico_R}" BARYCENTRIC "${resamp_R}"
    fi

done
