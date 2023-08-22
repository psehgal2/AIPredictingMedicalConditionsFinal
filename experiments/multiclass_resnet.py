import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets, TestDataset
from resnet_multi_condition import *

import argparse
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights
)

sys.stdout.write('Multiclass ResNet started.\n')
parser = argparse.ArgumentParser(
    description="ResNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-r", "--resnet", type=int)
args = parser.parse_args()
config = vars(args)
print(config)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
VISUALIZATION_HEAD = '/groups/CS156b/2023/yasers_beavers/visualizations'
CONDITION = [ # ALL CONDITIONS
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]
TRAIN_DOWNSAMPLE = config['downsample'] # 0.75
EPOCHS = config['epochs'] # 75
TRANSFORMS = []
RESNET, WEIGHTS = {18 : (resnet18, ResNet18_Weights.DEFAULT),
                   34 : (resnet34, ResNet34_Weights.DEFAULT),
                   50 : (resnet50, ResNet50_Weights.DEFAULT),
                   101 : (resnet101, ResNet101_Weights.DEFAULT),
                   152 : (resnet152, ResNet152_Weights.DEFAULT)}[config['resnet']]
device = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
ConditionData = BinaryPathologyDatasets(
    annotations=f'{DATA_HEAD}/student_labels/train2023.csv', 
    img_dir=DATA_HEAD, 
    condition=CONDITION,
    downsample=TRAIN_DOWNSAMPLE,
    transforms=TRANSFORMS,
    drop=False,
    mc=True
)
TestData = TestDataset(
    annotations=f'{DATA_HEAD}/student_labels/test_ids.csv', 
    img_dir=DATA_HEAD, 
    transforms=TRANSFORMS,
    cpu=(device == 'cpu')
)

TrainData = ConditionData.train
ValidData = ConditionData.valid
net = ResNetTrainMultiCondition(
    TrainData,
    ValidData,
    len(CONDITION),
    RESNET, # torchvision.models.resnet18 OR torchvision.models.resnet50
    WEIGHTS,
    EPOCHS, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_' \
    f'{"_".join([c.replace(" ", "_") for c in CONDITION])}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth',
    f'{VISUALIZATIONS_HEAD}/resnet{config["resnet"]}_' \
    f'{"_".join([c.replace(" ", "_") for c in CONDITION])}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.png'
)
ResNetPredictMultiCondition(
    TestData, 
    net, 
    device,
    CONDITION,
    TestData.labels. # TODO ,
    f'{PREDICTION_HEAD}/resnet{config["resnet"]}_' \
    f'{"_".join([c.replace(" ", "_") for c in CONDITION])}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.csv'
)
