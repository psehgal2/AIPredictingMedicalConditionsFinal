import numpy as np
import pandas as pd
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset_loaders import BinaryPathologyDatasets, TestDataset
from visualizers import ErrorPlotter
import segmentation_models_pytorch_maybe as smp
import segmentation_models_pytorch_maybe.utils as utils
# We imported stuff!
from torchvision.models import resnet50, ResNet50_Weights
class ActivatedUNet(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        ENCODER = 'resnet50'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = class_names
        ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

        self.model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
        self.activation = F.sigmoid
        self.onelayer = nn.Linear(288, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.onelayer(x)
        x = self.activation(x) 
        return x

def UNetTrainOneCondition(
    train_data, 
    valid_data, 
    epochs, # number of epochs for training
    model_path, # path to save the model to
    class_names
):
    # Build DataLoaders
    TrainLoader = DataLoader(train_data, batch_size=32, shuffle =True, drop_last=False)
    ValidLoader = DataLoader(valid_data, batch_size=32, shuffle=False, drop_last=False)
    loss = utils.losses.MSELoss()
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    model = ActivatedUNet(CLASSES)
# define metrics
    metrics = [
        utils.metrics.IoU(threshold=0.5),
    ]
    TRAINING = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set num of epochs
    EPOCHS = epochs
    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    # Set up Resnet with pretraining
    train_epoch = utils.train.TrainEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      optimizer=optimizer,
      device=DEVICE,
      verbose=True,)	
    
    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    print(valid_epoch.metrics)
    if TRAINING:
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(TrainLoader)
            valid_logs = valid_epoch.run(ValidLoader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            torch.save(model.state_dict(), f'/groups/CS156b/2023/yasers_beavers/models/unet/epoch_{class_names}_{i}.pth')
    return model
            # Save model if a better val IoU score is obtained
            # if best_iou_score < valid_logs['iou_score']:
            #     best_iou_score = valid_logs['iou_score']
            #     torch.save(model, './best_model.pth')
            #     print('Model saved!')
                             
    #test_epoch = utils.train.TestEpoch(
    #    model,
    #    loss=loss,
    #    metrics=metrics,
    #    device=DEVICE,
    #    verbose=True,
    #)
    #print(test_epoch.metrics)  
    #DATA_HEAD = '/groups/CS156b/data'
    #TestData = TestDataset(
    #annotations=f'{DATA_HEAD}/student_labels/test_ids.csv',
    #img_dir=DATA_HEAD,
    #transforms=TRANSFORMS,
    #cpu=(device == 'cpu'),
    #)
  
    #valid_logs = test_epoch.run(TestData)
    #print("Evaluation on Test Data: ")
    #print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    #print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")	

def UNetPredictOneCondition(test_data, net, device, condition, preds_path):
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
