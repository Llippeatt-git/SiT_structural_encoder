#!/bin/bash
#SBATCH -J trial_recon
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/batch/train_recon_final_%j.out
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/logs/batch/train_recon_final_%j.err
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem-per-cpu 12G
#SBATCH --cpus-per-task 8
#SBATCH -t 0-6:00:00  

cd /ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/SiT_structural_encoder/tools

python3 ./reconstruction_pipeline.py

done

# bash chpc_resample_sphere_fs_space.sh # if you are running this, itll take like 7 hours for ABCD (N>8,000). but only about 1GB max.
