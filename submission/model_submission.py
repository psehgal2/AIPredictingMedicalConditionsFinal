import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets, TestDataset
from resnet_one_condition import *
import torch

import argparse
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights
)

sys.stdout.write('Started!\n')
parser = argparse.ArgumentParser(
    description="ResNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-c", "--condition", type=str)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-r", "--resnet", type=int)
args = parser.parse_args()
config = vars(args)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
CONDITION = config['condition'] # 'Pneumonia'
TRAIN_DOWNSAMPLE = config['downsample'] # 0.75
EPOCHS = config['epochs'] # 75
TRANSFORMS = []
RESNET, WEIGHTS = {18 : (resnet18, ResNet18_Weights.DEFAULT),
                   34 : (resnet34, ResNet34_Weights.DEFAULT),
                   50 : (resnet50, ResNet50_Weights.DEFAULT),
                   101 : (resnet101, ResNet101_Weights.DEFAULT),
                   152 : (resnet152, ResNet152_Weights.DEFAULT)}[config['resnet']] 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
# OneCondititonData = BinaryPathologyDatasets(
#     annotations=f'{DATA_HEAD}/student_labels/train2023.csv', 
#     img_dir=DATA_HEAD, 
#     condition=CONDITION,
#     downsample=TRAIN_DOWNSAMPLE,
#     transforms=TRANSFORMS
# )

TestData = TestDataset(
    annotations=f'{DATA_HEAD}/student_labels/test_ids.csv', 
    img_dir=DATA_HEAD, 
    transforms=TRANSFORMS,
    cpu=(device == 'cpu'),
)

# TrainData = OneCondititonData.train
# ValidData = OneCondititonData.valid
# net = ResNetTrainOneCondition(
#     TrainData,
#     ValidData,
#     RESNET, # torchvision.models.resnet18 OR torchvision.models.resnet50
#     WEIGHTS,
#     EPOCHS, # number of epochs for training
#     device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
#     f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth',
# )

net = RESNET(weights=WEIGHTS).to(device)
num_ftrs = net.fc.in_features
net.fc = ActivatedResNet(num_ftrs).to(device)

state_dict = torch.load(
    f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth'
)

net.load_state_dict(state_dict)

ResNetPredictOneCondition(
    TestData,
    net,
    device,
    CONDITION,
    f'{PREDICTION_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.csv'
)
