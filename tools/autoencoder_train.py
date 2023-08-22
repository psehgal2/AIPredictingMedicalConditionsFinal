import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset_loaders import BinaryPathologyDatasets, TestDataset
from visualizers import ErrorPlotter
from autoencoder_architecture import Autoencoder

def TrainAutoencoder(
    train_data,
    valid_data,
    epochs, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path, # path to save the model to
    vis_path, # path to save visualizations to
    hidden_dim=1,
    latent_dim=30,
):
    # Build DataLoaders
    TrainLoader = DataLoader(train_data, batch_size=32)
    ValidLoader = DataLoader(valid_data, batch_size=32)

    # Set up model and optimizers
    net = Autoencoder(3, hidden_dim, latent_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.33)
    curr = time.time()
    best_train, best_valid = 1.0, 1.0
    train_losses, valid_losses = [], []

    # Training loop
    sys.stdout.write(f'Training on {len(train_data)} samples...')
    for epoch in range(epochs): # loop over the dataset multiple times
        tle, vle = [], []
        for i, data in enumerate(TrainLoader, 0):
            inputs = data.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(inputs, outputs)
            loss.backward()
            optimizer.step()

            # take statistics
            tle.append(loss.cpu().detach().numpy())
        train_losses.append(np.mean(tle))

        with torch.no_grad():
            for data in ValidLoader:
                inputs = data.to(device)
                outputs = net(inputs)
                vle.append(criterion(inputs, outputs).cpu().detach().numpy())
        valid_losses.append(np.mean(vle))
        # scheduler.step()

        if epoch == 0 or min(train_losses[:-1]) > train_losses[-1]:
            torch.save(net.state_dict(), f'{model_path}_TRAIN.pth')
        if epoch == 0 or min(valid_losses[:-1]) > valid_losses[-1]:
            torch.save(net.state_dict(), f'{model_path}_VALID.pth')

        plot_losses(epoch, train_losses, valid_losses, vis_path)

    sys.stdout.write(f'Time taken: {time.time() - curr}')
    sys.stdout.write('Finished Training')

    torch.save(net.state_dict(), model_path)
    return net

def plot_losses(epoch, train_losses, valid_losses, vis_path):
    sys.stdout.write(
        f'\nEPOCH {epoch:03}\t' \
        f"Train Loss = {train_losses[-1]:.6f}\t" \
        f"Valid Loss = {valid_losses[-1]:.6f}\t" \
    )
    plt.clf()
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.savefig(f'{vis_path}.png')