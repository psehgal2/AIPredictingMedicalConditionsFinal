import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets, TestDataset
from resnet_one_condition import *

import argparse
from torchvision.models import resnet18, resnet50

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
RESNET = resnet18 if config['resnet'] == 18 else resnet50
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
    cpu=(device == 'cpu')
)

# TrainData = OneCondititonData.train
# ValidData = OneCondititonData.valid
# net = ResNetTrainOneCondition(
#     TrainData,
#     ValidData,
#     RESNET, # torchvision.models.resnet18 OR torchvision.models.resnet50
#     EPOCHS, # number of epochs for training
#     device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
#     f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth'
# )

net = torch.load(
    f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth'
)
ResNetPredictOneCondition(
    TestData,
    net,
    device,
    CONDITION,
    f'{PREDICTION_HEAD}/resnet{config["resnet"]}_{CONDITION.replace(" ", "_")}' \
    f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.csv'
)
