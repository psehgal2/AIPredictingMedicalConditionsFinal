#!/bin/bash
# Above line is required!

# Name of job in SLURM queue
#SBATCH --job-name=resnet50

# Output and error log locations (captures stdout and stderr respectively). Defaults to your homedir.
#SBATCH --output=/home/aghatare/%j.out
#SBATCH --error=/home/aghatare/%j.err

# Account to charge this computation time to. THIS LINE IS ESSENTIAL.
#SBATCH -A CS156b

# Estimated time this job will take. A job exceeding this time will be killed.
# Required parameter!
#SBATCH -t 48:00:00

# Total number of concurrent srun tasks. Most people will not need this.
#SBATCH --ntasks=1

# Number of CPU threads for each task as defined above. Most people will be
# using a single task, so this is the total number of threads required.
#SBATCH --cpus-per-task=1

# Total amount of system RAM for all tasks. Specify units with M and G. 
#SBATCH --mem=32G

# Request a single Tesla P100 GPU
#SBATCH --gres=gpu:1

# Send status emails to an email
#SBATCH --mail-user=aghatare@caltech.edu

# Enable email notifications for changes to the job state
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load any modules you may need
module load cuda/12.0
module load gcc/11.2.0

# Setup a python environment
# source /groups/CS156b/conda_installs/njanwani/bin/
source activate /groups/CS156b/conda_installs/njanwani

# Do some cool things!
cd /central/groups/CS156b/2023/yasers_beavers/
python experiments/adishree_resnet_scriptable.py -c "Support Devices" \
-d 1.0 -e 40 -r 50