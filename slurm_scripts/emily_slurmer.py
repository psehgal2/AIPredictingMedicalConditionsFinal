import os, time
os.system('cd /central/groups/CS156b/2023/yasers_beavers/slurm_scripts')

# Job name
NAME =          'resnet50_mc'

# Conditions to 
CONDITIONS =    ['All']
# 'No Finding',
# 'Enlarged Cardiomediastinum',
# 'Cardiomegaly',
# 'Lung Opacity',
# 'Pneumonia',
# 'Pleural Effusion',
# 'Pleural Other',
# 'Fracture',
# 'Support Devices'

# Time to run
TIME =          '48:00:00'

# Directory of python file
DIRECTORY =     '/central/groups/CS156b/2023/yasers_beavers'

# Python file (enter any arguments you need here)
FILE =          'experiments/multiclass_resnet.py'

# Your HPC username
USERNAME =      'eechoe'

# num epochs to train
EPOCHS =        30

# Downsampling of train data
DOWNSAMPLE =    0.5

# Model to run
MODEL =         50

os.system(f'mkdir /central/groups/CS156b/2023/yasers_beavers/experiments/outputs/{NAME}')
for cond in CONDITIONS:
    # print(f'sbatch example.sbatch {NAME} {cond} {TIME} {DIRECTORY} {FILE}')
    sbatch_file = f"{DIRECTORY}/slurm_scripts/out/{NAME}_{cond.replace(' ', '_')}.sbatch"
    with open(sbatch_file, 'w') as file:
        file.write(
            f'''#!/bin/bash
#SBATCH --job-name={NAME}
#SBATCH --output=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/{NAME}/{cond.replace(" ", "_")}.out
#SBATCH --error=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/{NAME}/{cond.replace(" ", "_")}.err
#SBATCH --open-mode=truncate
#SBATCH -A CS156b
#SBATCH -t {TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user={USERNAME}@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# module load cuda/11.1
# module load gcc/11.2.0

source activate /groups/CS156b/conda_installs/njanwani
cd {DIRECTORY}
python -u {FILE + f' -c "{cond}" -r {MODEL} -e {EPOCHS} -d {DOWNSAMPLE}'}
            '''
        )
    os.system('chmod u+x ' + sbatch_file)
    os.system('chmod 777 ' + sbatch_file)
    time.sleep(0.25)
    os.system('sbatch ' + sbatch_file)


