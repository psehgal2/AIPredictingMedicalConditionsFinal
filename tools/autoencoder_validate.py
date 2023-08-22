import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loaders import ReconstructionDatasets
from autoencoder_architecture import Encoder, Autoencoder

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
TRAIN_DOWNSAMPLE = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'
ORIENTATIONS = ['Lateral', 'Frontal']

# Loaded model takes slightly different shape
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

# Model
net = ReAutoencoder(3, 1, 30).to(device)
net.load_state_dict(torch.load(
    f'{CHECKPOINT_HEAD}/autoenc_epochs_10_train_0.3_TRAIN.pth',
    map_location=torch.device('cpu')
))
criterion = nn.BCELoss()
sys.stdout.write('Loaded model successfully\n')

# Validate on subpopulations
validation_losses = {}
for ORIENTATION in ORIENTATIONS:
    Datasets = ReconstructionDatasets(
        annotations=f'/groups/CS156b/2023/yasers_beavers/data/train2023.csv', 
        img_dir=DATA_HEAD, 
        downsample=TRAIN_DOWNSAMPLE,
        transforms=[],
        orientation=ORIENTATION,
    )

    ValidLoader = DataLoader(Datasets.valid, batch_size=32)
    vle = []
    with torch.no_grad():
        for data in ValidLoader:
            inputs = data[0].to(device)
            outputs = torch.reshape(net(inputs), (3, 256, 256))
            vle.append(criterion(inputs, outputs).cpu().detach().numpy())
    validation_losses[ORIENTATION] = [np.mean(vle).round(4), np.std(vle).round(4)]

sys.stdout.write(validation_losses)