#!/bin/bash
#SBATCH --job-name=mvae_bottleneck
#SBATCH --partition=RAEAI
#SBATCH --gres=gpu:l40:3
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --output=/dataNAS/people/akkumar/Downloads/MedVAE/medvae/scripts/logs/bottleneck_%j.out
#SBATCH --error=/dataNAS/people/akkumar/Downloads/MedVAE/medvae/scripts/logs/bottleneck_%j.out

# Full-volume (224x224x160) image-space training of the 3D global bottleneck,
# model-parallel across the 3 allocated L40s (single process). Submit with:
#   sbatch medvae/scripts/train_bottleneck.sbatch
# Monitor: squeue -u $USER ; tail -f medvae/scripts/logs/bottleneck_<jobid>.out

set -euo pipefail
cd /dataNAS/people/akkumar/Downloads/MedVAE
mkdir -p medvae/scripts/logs

PY=/dataNAS/people/akkumar/Downloads/miniconda3/envs/medvae/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The 3 GPUs slurm allocated are already exposed as cuda:0,1,2 to this process;
# model_parallel_gpus=3 in the experiment config maps the decoder shards onto them.
srun "$PY" medvae/medvae_finetune.py \
    experiment=medvae_4x_1c_3d_bottleneck_finetuning
