#!/bin/bash
#SBATCH --job-name=validation
#SBATCH --output=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/reconstruction/pred_validation_BCE3.out
#SBATCH --error=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/reconstruction/pred_validation_BCE3.err
#SBATCH --open-mode=truncate
#SBATCH -A CS156b
#SBATCH -t 5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=aghatare@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# module load cuda/11.1
# module load gcc/11.2.0

source activate /groups/CS156b/conda_installs/njanwani
cd /central/groups/CS156b/2023/yasers_beavers
CUDA_LAUNCH_BLOCKING=1 python -u tools/au_predicter_validate.py
