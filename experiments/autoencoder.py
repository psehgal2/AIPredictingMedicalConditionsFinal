import argparse
import torch

import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import ReconstructionDatasets
from autoencoder_train import TrainAutoencoder

sys.stdout.write('Autoencoder started.\n')
parser = argparse.ArgumentParser(
    description="Autoencoder",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-l", "--latentdim", type=int)
parser.add_argument("-i", "--hiddendim", type=int)
args = parser.parse_args()
config = vars(args)
print(config)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
VISUALIZATION_HEAD = '/groups/CS156b/2023/yasers_beavers/visualizations2'
TRAIN_DOWNSAMPLE = config['downsample']
EPOCHS = config['epochs']
TRANSFORMS = []
LATENT_DIM = config['latentdim']
HIDDEN_DIM = config['hiddendim']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
Datasets = ReconstructionDatasets(
    annotations=f'{DATA_HEAD}/student_labels/train2023.csv', 
    img_dir=DATA_HEAD, 
    downsample=TRAIN_DOWNSAMPLE,
    transforms=TRANSFORMS,
)

TrainData = Datasets.train
ValidData = Datasets.valid
net = TrainAutoencoder(
    TrainData,
    ValidData,
    EPOCHS, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f'{CHECKPOINT_HEAD}/autoenc_DICE_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}_{LATENT_DIM}_{HIDDEN_DIM}',
    f'{VISUALIZATION_HEAD}/autoenc_DICE_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}_{LATENT_DIM}_{HIDDEN_DIM}',
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM,
)
