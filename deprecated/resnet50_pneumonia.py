import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import Resize, Grayscale, RandomHorizontalFlip, RandomCrop, ToTensor, Compose
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50
from PIL import Image
from visualizers import ErrorPlotter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kornia.filters import Sobel

head = '/groups/CS156b/data'
TRAIN_DOWNSAMPLE = 0.25
TEST_DOWNSAMPLE = 1.0
EPOCHS = 50

print('Imports complete')
print(f'Training on {100 * TRAIN_DOWNSAMPLE}% of data for {EPOCHS} epochs')
print(f'Predicting {100 * TEST_DOWNSAMPLE}% of data')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE =', device)

class LungDatasets():
    def __init__(self, annotations, img_dir, condition, downsample=1, train_split=0.8, transforms=[]):
        df = pd.read_csv(annotations).dropna(subset=condition).sample(frac=downsample)
        df = df[df[condition] != 0]
        df[condition] = df[condition].apply(lambda x : 0 if x == -1 else 1)
        is_train = np.random.rand(len(df)) < train_split
        self.train = LungDataset(df[is_train].reset_index(drop=True), img_dir, condition, transforms)
        self.valid = LungDataset(df[~is_train].reset_index(drop=True), img_dir, condition, transforms)

class LungDataset(Dataset):
    def __init__(self, label_df, img_dir, condition, transforms=[]):
        self.labels = label_df
        self.images = img_dir
        self.condition = condition
        self.transforms = Compose([
            Resize(250),
            RandomCrop(224),
            RandomHorizontalFlip()
        ])
        self.extra_transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(head + '/' + self.labels.loc[idx]['Path'], ImageReadMode.RGB)
        # image = Image.open(head + '/' + self.labels.loc[idx]['Path']).convert('RGB')
        if 'sobel' in self.extra_transforms:
            image = Sobel()(image)
        label = self.labels.loc[idx][self.condition]
        return self.transforms(image), label

class TestDataset(Dataset):
    def __init__(self, annotations, img_dir, downsample=1.0):
        if TEST_DOWNSAMPLE < 1.0:
            self.labels = pd.read_csv(annotations).sample(frac=downsample).reset_index(drop=True)
        else:
            self.labels = pd.read_csv(annotations)
        self.images = img_dir
        self.transforms = Compose([
            Resize(250),
            RandomCrop(224),
            RandomHorizontalFlip()
        ])
        self.extra_transforms = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(head + '/' + self.labels.loc[idx]['Path'], ImageReadMode.RGB)
        # image = Image.open(head + '/' + self.labels.loc[idx]['Path']).convert('RGB')
        if 'sobel' in self.extra_transforms:
            image = Sobel()(image)

        return self.transforms(image)
    
class Net(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.model = nn.Linear(in_features,1)
        self.activation = F.sigmoid

    def forward(self, x):
        # print(x)
        x = self.model(x)
        x = self.activation(x)
        return x

LungData = LungDatasets(head + '/student_labels/train2023.csv', head, 'Pneumonia', downsample=TRAIN_DOWNSAMPLE)
TrainData = LungData.train
ValidData = LungData.valid
TestData = TestDataset(head + '/student_labels/test_ids.csv', head, downsample=TEST_DOWNSAMPLE)

TrainLoader = DataLoader(TrainData, batch_size=32)
ValidLoader = DataLoader(ValidData, batch_size=32)
TestLoader = DataLoader(TestData, batch_size=len(TestData))

net = resnet18(pretrained=True).to(device) #Net().to(device)
num_ftrs = net.fc.in_features
net.fc = Net(num_ftrs).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
curr = time.time()
train_losses, valid_losses, train_accuracy, test_accuracy = [[]] * 4
EP = ErrorPlotter()

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    train_loss, train_acc, valid_loss, valid_acc = [0] * 4
    train_total, valid_total = 0,0
    for i, data in enumerate(TrainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())[:,0]
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # take statistics
        train_loss += loss.item()
        train_acc += ((outputs > 0.5) == (labels > 0.5)).sum().item()
        train_total += len(outputs)

    with torch.no_grad():
        for data in ValidLoader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images.float())

            # get statistics
            valid_total += len(labels)
            valid_acc += ((outputs > 0.5) == (labels > 0.5))[0].sum().item()
            valid_loss += criterion(outputs[:,0], labels.float()).item()
    
    print(f'EPOCH {epoch:03}\
            Train Acc = {train_acc / train_total:.3f}\
            Train Loss = {train_loss / train_total:.3f}\
            Valid Acc = {valid_acc / valid_total:.3f}\
            Valid Loss = {valid_loss / valid_total:.3f}')

    EP.update(
        [train_loss / train_total, valid_loss / valid_total],
        [train_acc / train_total, valid_acc / valid_total]
    )

print('Time taken', time.time() - curr)
print('Finished Training')

EP.plot()

outputs = None
for d in TestLoader:
   outputs = np.array(net(d.to(device).float()).cpu().detach())

#    print(outputs)
#    print(len(outputs))

pred_df = pd.DataFrame()
pred_df['Pneumonia'] = outputs[:,0]
now = datetime.now()
torch.save(net.state_dict(), f'/groups/CS156b/2023/yasers_beavers/checkpoints/{now.strftime("%d_%m_%Y_%H_%M_%S")}_pneumonia.pth')
pred_df.to_csv(fr'/central/groups/CS156b/2023/yasers_beavers/data/{now.strftime("%d_%m_%Y_%H_%M_%S")} _predictions.csv')
