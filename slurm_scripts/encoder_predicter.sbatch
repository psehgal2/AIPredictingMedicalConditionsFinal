#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --output=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/reconstruction/train_prediction_Pneumonia_05_28_IND_f1.out
#SBATCH --error=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/reconstruction/train_prediction_Pneumonia_05_28_IND_f1.err
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
python -u experiments/encoder_predicter.py -e 30 -d 1 -c Pneumonia -f 1
