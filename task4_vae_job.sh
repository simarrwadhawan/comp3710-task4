#!/bin/bash
#SBATCH --job-name=task4_vae
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=task4_vae_%j.out

module purge
source ~/miniconda3/bin/activate dawnbench

python task4_vae.py

