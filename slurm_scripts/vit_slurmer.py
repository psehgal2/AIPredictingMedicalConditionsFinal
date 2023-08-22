import os, time
os.system('cd /central/groups/CS156b/2023/yasers_beavers/slurm_scripts')

# Job name
NAME =          'vit'

# Conditions to 
CONDITIONS =    ['No Finding',
                 'Enlarged Cardiomediastinum',
                 'Cardiomegaly',
                 'Lung Opacity',
                 'Pneumonia',
                 'Pleural Other',
                 'Pleural Effusion',
                 'Fracture',
                 'Support Devices']

# CONDITIONS = ['Fracture']
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
FILE =          'experiments/vit_scriptable.py'

# Your HPC username
USERNAME =      'njanwani'

# num epochs to train
EPOCHS =        {cond : 20 for cond in CONDITIONS}

# Downsampling of train data
DOWNSAMPLE =    {cond : 0.6 for cond in CONDITIONS}
DOWNSAMPLE['Pleural Effusion'] = 0.4
DOWNSAMPLE['No Finding'] = 0.4

# Model to run
MODEL =         {cond : 'b16' for cond in CONDITIONS}

# Batch sizes
BATCH_SIZE =    {cond : 64 for cond in CONDITIONS}
BATCH_SIZE['Pleural Effusion'] = 64
BATCH_SIZE['No Finding'] = 64
BATCH_SIZE['Pleural Other'] = 8
# BATCH_SIZE['Fracture'] = 1

# Images resizing
IMAGE_RESIZE =  {cond : 224 for cond in CONDITIONS}
# IMAGE_RESIZE['Fracture'] = 1000

os.system(f'mkdir /central/groups/CS156b/2023/yasers_beavers/experiments/outputs/{NAME}')
for cond in CONDITIONS:
    # print(f'sbatch example.sbatch {NAME} {cond} {TIME} {DIRECTORY} {FILE}')
    sbatch_file = f"{DIRECTORY}/slurm_scripts/out/{NAME}_{cond.replace(' ', '_')}.sbatch"
    with open(sbatch_file, 'w') as file:
        file.write(
            f'''#!/bin/bash
#SBATCH --job-name={NAME}-{cond.replace(" ","_")}
#SBATCH --output=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/{NAME}/{MODEL[cond]}_{cond.replace(" ", "_")}.out
#SBATCH --error=/central/groups/CS156b/2023/yasers_beavers/experiments/outputs/{NAME}/{MODEL[cond]}_{cond.replace(" ", "_")}.err
#SBATCH --open-mode=truncate
#SBATCH -A CS156b
#SBATCH -t {TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --mail-user={USERNAME}@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# module load cuda/11.1
# module load gcc/11.2.0

source activate /groups/CS156b/conda_installs/njanwani
cd {DIRECTORY}
python -u {FILE + f' -c "{cond}" -v {MODEL[cond]} -e {EPOCHS[cond]} -d {DOWNSAMPLE[cond]} -b {BATCH_SIZE[cond]} -s {IMAGE_RESIZE[cond]}'}
            '''
        )
    os.system('chmod u+x ' + sbatch_file)
    os.system('chmod 777 ' + sbatch_file)
    time.sleep(0.25)
    os.system('sbatch ' + sbatch_file)


# python -u {FILE + f' -c "{cond}" -r {MODEL} -e {EPOCHS} -d {DOWNSAMPLE}'}