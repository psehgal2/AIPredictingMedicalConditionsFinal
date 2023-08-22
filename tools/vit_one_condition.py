import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset_loaders import BinaryPathologyDatasets, TestDataset
from visualizers import ErrorPlotter

class ActivatedViT(nn.Module):
    def __init__(self, vit, weights, dropout=0.0):
        super().__init__()
        self.vit = vit(weights=weights)
        self.out_layer = nn.Linear(1000, 1)
        self.activation = F.sigmoid
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.vit(x)
        x = self.out_layer(x)
        # x = self.dropout(x)
        x = self.activation(x)
        return x

def ViTTrainOneCondition(
    train_data, 
    valid_data, 
    vit, # torchvision.models.resnet18 OR torchvision.models.resnet50
    weights,
    epochs, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path, # path to save the model to
    condition, # condition to predict
    batch_size # batch_size
):
    # Build DataLoaders
    TrainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    ValidLoader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Set up Resnet with pretraining
    # net = vit().to(device)
    net = ActivatedViT(vit, weights).to(device)
    # num_ftrs = net.fc.in_features
    # net.fc = ActivatedResNet(num_ftrs).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    curr = time.time()
    best_train, best_valid = 1.0, 1.0
    EP = ErrorPlotter()

    # Training loop
    for epoch in range(epochs):  # loop over the dataset multiple times
        EP.start_epoch()
        for i, data in enumerate(TrainLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            # print(outputs.shape)
            # print(labels)
            # print()
            # print()
            loss = criterion(outputs[:,0], labels.float())
            loss.backward()
            optimizer.step()

            # take statistics
            EP.update_train(loss, outputs[:,0], labels)
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

        

    print('Time taken', time.time() - curr)
    print('Finished Training')

    EP.plot(vis_path=f'/groups/CS156b/2023/yasers_beavers/visualizations/vit/{condition}',
            plot_title=f'{vit.__name__} {condition}')
    torch.save(net.state_dict(), f'{model_path}_FINAL')
    return net

def ViTPredictOneCondition(test_data, net, device, condition, preds_path):
    TestLoader = DataLoader(test_data, batch_size=1)

    outputs = np.array([])
    curr = time.time()
    for idx, d in enumerate(TestLoader):
        if idx % 100 == 9:
            curr = time.time()
        outputs = np.append(outputs, np.array(net(d.to(device).float()).cpu().detach()))

    pred_df = pd.DataFrame()
    pred_df[condition] = outputs
    pred_df.to_csv(preds_path)
