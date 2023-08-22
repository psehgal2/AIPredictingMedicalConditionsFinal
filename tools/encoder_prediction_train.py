import sys
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset_loaders import BinaryPathologyDatasets, TestDataset
from resnet_one_condition import ActivatedResNet
from autoencoder_architecture import Encoder, Autoencoder
from visualizers import ErrorPlotter

class EncoderPredicter(nn.Module):
    def __init__(self, autoencoder, fix_enc=True, latent_dim=30, dropout=0.2, 
        nonlinearity=nn.Sigmoid):
        super().__init__()
        self.encoder = autoencoder.encoder
        if fix_enc: # freeze encoder layers
            for param in self.encoder.parameters(): param.requires_grad = False
        # Configure the prediction layers
        layer = lambda in_dim, out_dim: nn.Sequential(
            nn.Linear(in_dim, out_dim), nonlinearity(), nn.Dropout(dropout)
        )
        self.predicter = nn.Sequential(
            layer(latent_dim, 20), layer(20, 20), layer(20, 10),
            nn.Linear(10, 1), nonlinearity(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.predicter(x)
        return x

class InverseEncoderPredicter(nn.Module):
    def __init__(self, autoencoder, resnet, weights,
        fix_enc=True, latent_dim=30, dropout=0.2, nonlinearity=nn.Sigmoid):
        super().__init__()
        self.autoencoder = autoencoder
        if fix_enc: # freeze encoder layers
            for param in self.autoencoder.parameters(): param.requires_grad = False
        # Configure prediction layers
        self.predicter = resnet(weights=weights)
        num_ftrs = self.predicter.fc.in_features
        self.predicter.fc = ActivatedResNet(num_ftrs, dropout=0)

    def forward(self, x):
        r = self.autoencoder(x)
        x = self.predicter(r - x)
        return x

def EncoderPredicterTrain(
    train_data, 
    valid_data, 
    condition, # condition to predict
    autoencoder_path,
    epochs, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path, # path to save the model to
    vis_path,
    batch_size, # batch_size
    latent_dim=30,
    hidden_dim=1,
    fix_enc=True,
    clf='direct', # 'direct' or 'inverse'
    resnet=None,
    weights=None,
):
    # Build DataLoaders
    TrainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    ValidLoader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Set up model
    autoencoder = Autoencoder(3, hidden_dim, latent_dim)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    if clf == 'direct':
        net = EncoderPredicter(autoencoder, fix_enc, latent_dim).to(device)
    else:
        assert(resnet is not None and weights is not None)
        net = InverseEncoderPredicter(
            autoencoder, resnet, weights, fix_enc, latent_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    curr = time.time()
    best_train, best_valid = 1.0, 1.0
    EP = ErrorPlotter()

    # Training loop
    for epoch in range(epochs):  # loop over the dataset multiple times
        EP.start_epoch()
        for i, data in enumerate(TrainLoader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs / 255)[:,0]
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            EP.update_train(loss, outputs, labels)
        scheduler.step()

        with torch.no_grad():
            for data in ValidLoader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images.float())[:,0]
                EP.update_valid(criterion, outputs, labels)

        EP.finish_epoch(epoch)

        if best_train > EP.get_train():
            torch.save(net.state_dict(), f'{model_path}_TRAIN')
        if best_valid > EP.get_valid():
            torch.save(net.state_dict(), f'{model_path}_VALID')

        EP.plot(vis_path, f'MSE loss for predicting {condition}')
        
    sys.stdout.write(f'Time taken {time.time() - curr}')
    sys.stdout.write('Finished Training')

    EP.plot(vis_path, f'MSE loss for predicting {condition}')
    torch.save(net.state_dict(), f'{model_path}_FINAL')
    return net
