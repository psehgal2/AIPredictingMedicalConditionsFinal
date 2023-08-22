import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import MultiPathologyDatasets, TestDataset
from resnet_one_condition import ResNetTrainOneCondition

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
    description="Hierarchical model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-d", "--downsample", type=float)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-r", "--resnet", type=int)
args = parser.parse_args()
config = vars(args)

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions/hierarchical'
CONDITION_SET = [
    ['Enlarged Cardiomediastinum', 'Cardiomegaly'],
    ['Pleural Other', 'Pleural Effusion'],
    ['Lung Opacity', 'Pneumonia'], 
    ['Fracture', 'Support Devices']
]
TRAIN_DOWNSAMPLE = config['downsample'] # 0.75
EPOCHS = config['epochs'] # number of epochs for training, e.g., 75 
TRANSFORMS = []
RESNET, WEIGHTS = {18 : (resnet18, ResNet18_Weights), # pretrained model to use
                   34 : (resnet34, ResNet34_Weights),
                   50 : (resnet50, ResNet50_Weights),
                   101 : (resnet101, ResNet101_Weights),
                   152 : (resnet152, ResNet152_Weights)}[config['resnet']] 
device = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Layer 1 models
for category in CONDITION_SET:
    CategoryData = MultiPathologyDatasets(
        annotations=f'{DATA_HEAD}/student_labels/train2023.csv', 
        img_dir=DATA_HEAD, 
        condition=category,
        downsample=TRAIN_DOWNSAMPLE,
        transforms=TRANSFORMS,
        drop=True,
        mc=True,
    )

    TrainData = CategoryData.train
    ValidData = CategoryData.valid

    net = ResNetTrainOneCondition(
        TrainData,
        ValidData,
        RESNET,
        WEIGHTS,
        EPOCHS,
        device,
        f'{CHECKPOINT_HEAD}/resnet{config["resnet"]}_' \
        f'{"_".join([c.replace(" ", "_") for c in TrainData.condition])}' \
        f'_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth'
    )

