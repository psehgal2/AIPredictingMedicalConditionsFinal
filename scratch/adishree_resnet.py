import sys
sys.path.append('/groups/CS156b/2023/yasers_beavers/tools')
from dataset_loaders import BinaryPathologyDatasets, TestDataset
from resnet_one_condition import *
from torchvision.models import resnet50

DATA_HEAD = '/groups/CS156b/data'
CHECKPOINT_HEAD = '/groups/CS156b/2023/yasers_beavers/checkpoints'
PREDICTION_HEAD = '/groups/CS156b/2023/yasers_beavers/predictions'
CONDITION = 'Pneumonia'
TRAIN_DOWNSAMPLE = 0.75
EPOCHS = 75
TRANSFORMS = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 'cpu'

# Collect data
OneCondititonData = BinaryPathologyDatasets(
    annotations=f'{DATA_HEAD}/student_labels/train2023.csv', 
    img_dir=DATA_HEAD, 
    condition=CONDITION,
    downsample=TRAIN_DOWNSAMPLE,
    transforms=TRANSFORMS
)
TestData = TestDataset(
    annotations=f'{DATA_HEAD}/student_labels/test_ids.csv', 
    img_dir=DATA_HEAD, 
    transforms=TRANSFORMS,
    cpu=(device == 'cpu')
)

TrainData = OneCondititonData.train
ValidData = OneCondititonData.valid
net = ResNetTrainOneCondition(
    TrainData,
    ValidData,
    resnet50, # torchvision.models.resnet18 OR torchvision.models.resnet50
    EPOCHS, # number of epochs for training
    device, # 'cpu' OR torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f'{CHECKPOINT_HEAD}/resnet_50_{CONDITION}_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.pth'
)
ResNetPredictOneCondition(
    TestData, 
    net, 
    device, 
    CONDITION, 
    f'{PREDICTION_HEAD}/resnet_50_{CONDITION}_epochs_{EPOCHS}_train_{TRAIN_DOWNSAMPLE}.csv'
)
