import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets, TestDataset
from unet_one_condition import *

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
    description="UNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-c", "--condition", type=str)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-u", "--unet", type=int)
args = parser.parse_args()
config = vars(args)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
CONDITION = config['condition'] # 'Pneumonia'
TRAIN_DOWNSAMPLE = config['downsample'] # 0.75
EPOCHS = config['epochs'] # 75
TRANSFORMS = []
device = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
OneCondititonData = BinaryPathologyDatasets(
    annotations=f'{DATA_HEAD}/student_labels/train2023.csv', 
    img_dir=DATA_HEAD, 
    condition=CONDITION,
    downsample=TRAIN_DOWNSAMPLE,
    transforms=TRANSFORMS,
    size=288
)

TestData = TestDataset(
    annotations=f'{DATA_HEAD}/student_labels/test_ids.csv', 
    img_dir=DATA_HEAD, 
    transforms=TRANSFORMS,
    cpu=(device == 'cpu'),
    size=288
)

TrainData = OneCondititonData.train
ValidData = OneCondititonData.valid
net = UNetTrainOneCondition(
    TrainData,
    ValidData,
    EPOCHS, # number of epochs for training
    f'{CHECKPOINT_HEAD}/unet{config["unet"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth',   
    CONDITION 
)

UNetPredictOneCondition(
    TestData,
    net,
    device,
    CONDITION,
    f'{PREDICTION_HEAD}/unet{config["unet"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.csv'
)
