import argparse
import torch

from torchvision.models import resnet50, ResNet50_Weights

import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets
from encoder_prediction_train import EncoderPredicterTrain

# Collect arguments
sys.stdout.write('Encoder-Predictor started.\n')
parser = argparse.ArgumentParser(
    description="Encoder-Predictor",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-c", "--condition", type=str)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-f", "--fixencoding", type=int)
args = parser.parse_args()
config = vars(args)
print(config)

# Set up variables
DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
VISUALIZATION_HEAD = '/groups/CS156b/2023/yasers_beavers/visualizations2'
CONDITION = config['condition'] # 'Pneumonia'
TRAIN_DOWNSAMPLE = config['downsample'] # 0.75
EPOCHS = config['epochs'] 
FIX_ENC = config['fixencoding'] # 0 (False) or 1 (True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
OneCondititonData = BinaryPathologyDatasets(
    annotations=f'/groups/CS156b/2023/yasers_beavers/data/train2023.csv', 
    img_dir=DATA_HEAD,
    condition=CONDITION,
    downsample=TRAIN_DOWNSAMPLE,
    transforms=[],
    size=256,
    crop=1
)
net = EncoderPredicterTrain(
    OneCondititonData.train,
    OneCondititonData.valid,
    CONDITION,
    # f'{CHECKPOINT_HEAD}/autoenc_epochs_10_train_0.3_60_2',
    f'{CHECKPOINT_HEAD}/autoenc_epochs_10_train_0.3_TRAIN.pth',
        # autoencoder encoder model after 2 epochs of training
    EPOCHS,
    device,
    f'{CHECKPOINT_HEAD}/ep_ind_epochs_{EPOCHS}_{CONDITION.replace(" ", "_")}_' \
    f'downsample_{TRAIN_DOWNSAMPLE}_FIX_ENC_{FIX_ENC}.pth',
    f'{VISUALIZATION_HEAD}/ep_ind_epochs_{EPOCHS}_{CONDITION.replace(" ", "_")}_' \
    f'downsample_{TRAIN_DOWNSAMPLE}_FIX_ENC_{FIX_ENC}.png',
    32,
    fix_enc=FIX_ENC,
    hidden_dim=1,
    latent_dim=30,
    clf='indirect',
    resnet=resnet50,
    weights=ResNet50_Weights.DEFAULT,
)