import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import Resize, Grayscale, RandomHorizontalFlip, RandomCrop
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet18 as resnet18

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

head = '/groups/CS156b/data'
TRAIN_DOWNSAMPLE = 0.01
TEST_DOWNSAMPLE = 1.0
EPOCHS = 1

print('Imports complete')
print(f'Training on {100 * TRAIN_DOWNSAMPLE}% of data for {EPOCHS} epochs')
print(f'Predicting {100 * TEST_DOWNSAMPLE}% of data')
device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE =', device)

class LungDatasets():
    def __init__(self, annotations, img_dir, condition, downsample=1, train_split=0.8):
        df = pd.read_csv(annotations).dropna(subset=condition).sample(frac=downsample)
        df = df[df[condition] != 0]
        df[condition] = df[condition].apply(lambda x : 0 if x == -1 else 1)
        is_train = np.random.rand(len(df)) < train_split
        self.train = LungDataset(df[is_train].reset_index(drop=True), img_dir, condition)
        self.valid = LungDataset(df[~is_train].reset_index(drop=True), img_dir, condition)

class LungDataset(Dataset):
    def __init__(self, label_df, img_dir, condition):
        self.labels = label_df
        self.images = img_dir
        self.condition = condition
        self.transforms = torch.nn.Sequential(
            Resize(250),
            RandomCrop(224),
            RandomHorizontalFlip()
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(head + '/' + self.labels.loc[idx]['Path'], ImageReadMode.RGB)
        label = self.labels.loc[idx][self.condition]
        return self.transforms(image), label

class TestDataset(Dataset):
    def __init__(self, annotations, img_dir, downsample=1.0):
        self.labels = pd.read_csv(annotations).sample(frac=downsample).reset_index(drop=True)
        self.images = img_dir
        self.transforms = torch.nn.Sequential(
            Grayscale(1),
            Resize(250),
            RandomCrop(224),
            RandomHorizontalFlip()
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(head + '/' + self.labels.loc[idx]['Path'], ImageReadMode.RGB)
        return self.transforms(image)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 50)
        self.fc2 = nn.Linear(50, 7)
        self.fc3 = nn.Linear(7, 1)

    def forward(self, x):
        # print(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

LungData = LungDatasets(head + '/student_labels/train2023.csv', head, 'Pneumonia', downsample=TRAIN_DOWNSAMPLE)
TrainData = LungData.train
ValidData = LungData.valid
TestData = TestDataset(head + '/student_labels/test_ids.csv', head, downsample=TEST_DOWNSAMPLE)

TrainLoader = DataLoader(TrainData, batch_size=32)
ValidLoader = DataLoader(ValidData, batch_size=32)
TestLoader = DataLoader(TestData, batch_size=len(TestData))

net = resnet18(pretrained=True) #Net().to(device)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
curr = time.time()
train_losses, valid_losses, train_accuracy, test_accuracy = [[]] * 4

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    train_loss, train_acc, valid_loss, valid_acc = [0] * 4
    train_total, valid_total = 0,0
    for i, data in enumerate(TrainLoader, 0):
        print('Shape:', data[0].shape)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print("Outputs:", outputs.shape)
        print("Labels:",labels.shape)

        # print statistics
        train_loss += loss.item()
        # print(outputs)
        train_acc += ((outputs > 0.5) == (labels > 0.5)).sum().item()
        train_total += len(outputs)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in ValidLoader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images.float())
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1) <-- Neil: we may need this for multiclass prediction but not for binary classifier
            valid_total += len(labels)
            # valid_acc += ((outputs > 0.5) == (labels > 0.5))[0].sum().item()

            valid_loss += criterion(outputs[:,0], labels.float()).item()

    # print(f'EPOCH {epoch}: Accuracy of the network on the {valid_total} validation images: {100 * valid_acc / valid_:.3f}%')
    # print(f'Training loss: {running_loss / i:.3f}')
    # print(f'Validation loss: {valid_loss / total:.3f}')
    print(f'EPOCH {epoch:03}\
            Train Acc = {train_acc / train_total:.3f}\
            Train Loss = {train_loss / train_total:.3f}\
            Valid Acc = {valid_acc / valid_total:.3f}\
            Valid Loss = {valid_loss / valid_total:.3f}')
    # print('---')

print('Time taken', time.time() - curr)

print('Finished Training')

torch.save(net.state_dict(), '/groups/CS156b/2023/yasers_beavers/checkpoints/04-22-0_pneumonia.pth')

outputs = None
for d in TestLoader:
   outputs = np.array(net(d.to(device).float()).cpu().detach())

   print(outputs)
   print(len(outputs))

pred_df = pd.DataFrame()
pred_df['Pneumonia'] = outputs[:,0]
pred_df.to_csv(r'/central/groups/CS156b/2023/yasers_beavers/data/pneuomia_predictions.csv')

# with open('predictions.csv', 'w') as file:
#     writer = csv.writer(file, lineterminator='\n')
#     for row in outputs:
#         writer.writerow(row)

# print(time.time() - curr)
# print('Predicted time:', len(TestData) / len(outputs) * (time.time() - curr))