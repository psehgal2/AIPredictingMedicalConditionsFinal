from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

import sys
sys.path.append('tools')
from dataset_loaders import ReconstructionDatasets
from autoencoder_architecture import Encoder, Autoencoder

class ReEncoder(Encoder):
    def forward(self, x):
        for i in range(5):
            x = self.net[i](x)
        x = torch.reshape(x, (1, -1))
        x = self.net[6](x)
        return x

class ReAutoencoder(Autoencoder):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__(in_channels, hidden_dim, latent_dim)
        self.encoder = ReEncoder(in_channels, hidden_dim, latent_dim)

CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
DATA_HEAD = '/groups/CS156b/data'
TRAIN_DOWNSAMPLE = 0.5

net = ReAutoencoder(3, 1, 30).to('cpu')
net.load_state_dict(torch.load(
    f'{CHECKPOINT_HEAD}/autoenc_epochs_10_train_0.3_TRAIN.pth',
    map_location=torch.device('cpu')
))

Datasets = ReconstructionDatasets(
    annotations=f'/groups/CS156b/2023/yasers_beavers/data/train2023.csv', 
    img_dir=DATA_HEAD, downsample=TRAIN_DOWNSAMPLE, transforms=[],
)
dt = Datasets.train

a = lambda x: Datasets.train.annotations[Datasets.train.annotations[x] == 1]
# a('Pneumonia').reset_index().loc[0]
img = lambda x, y: Image.fromarray(y.numpy() * 255).convert("L").save(
    f'visualizations2/example_images/{x}')

def draw(idx):
    img(f'dt_{idx}.png', dt[idx][0])
    img(f'dt_{idx}p.png', net(dt[idx]).detach()[0][0])

draw(0)
draw(10)