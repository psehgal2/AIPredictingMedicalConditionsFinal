import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets, TestDataset
from vit_one_condition import *
import torch

import argparse
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_b_32,
    ViT_B_32_Weights,
    vit_l_16,
    ViT_L_16_Weights,
    vit_l_32,
    ViT_L_32_Weights,
    vit_h_14,
    ViT_H_14_Weights,
)

sys.stdout.write('Started!\n')
parser = argparse.ArgumentParser(
    description="ResNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-c", "--condition", type=str)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-v", "--vit", type=str)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-s", "--image_resize", type=int)
args = parser.parse_args()
config = vars(args)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
CONDITION = config['condition'] # 'Pneumonia'
TRAIN_DOWNSAMPLE = config['downsample'] # 0.75
EPOCHS = config['epochs'] # 75
BATCH_SIZE = config['batch_size']
IMAGE_RESIZE = config['image_resize'] # usually 250
TRANSFORMS = []
VIT, WEIGHTS = {'b16' : (vit_b_16, ViT_B_16_Weights.DEFAULT),
                'b32' : (vit_b_32, ViT_B_32_Weights.DEFAULT),
                'l16' : (vit_l_16, ViT_L_16_Weights.DEFAULT),
                'l32' : (vit_l_32, ViT_L_32_Weights.DEFAULT),
                'h14' : (vit_h_14, ViT_H_14_Weights.DEFAULT)}[config['vit']] 
# VIT = vit_b_16
WEIGHTS = None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
OneCondititonData = BinaryPathologyDatasets(
    annotations=f'/groups/CS156b/2023/yasers_beavers/data/train2023.csv', 
    img_dir=DATA_HEAD, 
    condition=CONDITION,
    downsample=TRAIN_DOWNSAMPLE,
    transforms=TRANSFORMS,
    size=IMAGE_RESIZE
)

TestData = TestDataset(
    annotations=f'{DATA_HEAD}/student_labels/test_ids.csv', 
    img_dir=DATA_HEAD, 
    transforms=TRANSFORMS,
    cpu=(device == 'cpu'),
    size=IMAGE_RESIZE
)

TrainData = OneCondititonData.train
ValidData = OneCondititonData.valid
net = ViTTrainOneCondition(
    TrainData,
    ValidData,
    VIT, # torchvision.models.resnet18 OR torchvision.models.resnet50
    WEIGHTS,
    EPOCHS, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f'{CHECKPOINT_HEAD}/vit{config["vit"]}/{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth',
    CONDITION,
    BATCH_SIZE
)

# net = torch.load(
#     f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
#     f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth'
# )
ViTPredictOneCondition(
    TestData,
    net,
    device,
    CONDITION,
    f'{PREDICTION_HEAD}/vit{config["vit"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.csv'
)
