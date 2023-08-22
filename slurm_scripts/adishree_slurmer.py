import os, time
os.system('cd /central/groups/CS156b/2023/yasers_beavers/slurm_scripts')

NAME = 'inverse_encoder_predictor2'
CONDITIONS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    # 'Pneumonia',
    'Pleural Effusion',
    # 'Pleural Other',
    # 'Fracture',
    'Support Devices'
]
TIME = '48:00:00'
DIRECTORY = '/groups/CS156b/2023/yasers_beavers/'
FILE = 'experiments/encoder_predicter.py'
USERNAME = 'aghatare'
EPOCHS = 29
DOWNSAMPLE = 1
FIX_ENC = 1
os.system(f'mkdir {DIRECTORY}/experiments/outputs/{NAME}')
for cond in CONDITIONS:
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
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user={USERNAME}@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source activate /groups/CS156b/conda_installs/njanwani
cd {DIRECTORY}
python -u {FILE + f' -c "{cond}" -e {EPOCHS} -d {DOWNSAMPLE} -f {FIX_ENC}'}
'''
        )
    os.system('chmod u+x ' + sbatch_file)
    os.system('chmod 777 ' + sbatch_file)
    time.sleep(0.25)
    os.system('sbatch ' + sbatch_file)


