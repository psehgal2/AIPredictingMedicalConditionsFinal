from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from dataset_loaders import BinaryPathologyDatasets
from encoder_prediction_train import InverseEncoderPredicter
from autoencoder_validate import ReAutoencoder

print('Imports complete', file=sys.stdout)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
VISUALIZATION_HEAD = '/groups/CS156b/2023/yasers_beavers/visualizations2/example_images'
TRAIN_DOWNSAMPLE = 0.05
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Model
autoencoder = ReAutoencoder(3, 1, 30)
autoencoder.load_state_dict(torch.load(
    f'{CHECKPOINT_HEAD}/autoenc_epochs_10_train_0.3_TRAIN.pth', 
    # map_location=torch.device('cpu')
))
net = InverseEncoderPredicter(autoencoder, resnet50, ResNet50_Weights.DEFAULT,
        fix_enc=True, latent_dim=30, dropout=0.2, nonlinearity=nn.Sigmoid)
net.load_state_dict(torch.load(
    f'{CHECKPOINT_HEAD}/ep_ind_epochs_20_Pneumonia_downsample_1.0_FIX_ENC_1.pth_FINAL',
    # map_location=torch.device('cpu')
))

print('Loaded models', file=sys.stdout)

criterion = nn.BCELoss()

# Validate on subpopulations
validation_losses = {}
Datasets = BinaryPathologyDatasets(
    annotations=f'/groups/CS156b/2023/yasers_beavers/data/train2023.csv', 
    img_dir=DATA_HEAD, 
    downsample=TRAIN_DOWNSAMPLE,
    transforms=[],
)

print('Created datasets', file=sys.stdout)

ValidLoader = DataLoader(Datasets.valid, batch_size=32)
vle = []
cm = [0, 0, 0, 0]
with torch.no_grad():
    for data in ValidLoader:
        images, labels = data[0].float().to(device), data[1].float().to(device)
        outputs = net(images)[:,0]
        vle.append(criterion(outputs, labels).item())
        cm = np.add(cm, [confusion_matrix(labels, outputs).ravel()])
cm = cm / len(vle)
print(cm, file=sys.stdout)
print(np.mean(vle), file=sys.stdout)