import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
import cv2
import sys

head = '/central/groups/CS156b/data'

labels = pd.read_csv('/groups/CS156b/2023/yasers_beavers/data/train2023.csv')
transforms = torch.nn.Sequential(
            Resize(250, antialias=True)
            # RandomCrop(224),
            # RandomHorizontalFlip()
        )


def sobelfilter(imgpath):
    sys.path.append('/usr/local/lib/python3/site-packages')
    img = cv2.imread(imgpath) 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=img_gray, threshold1=50, threshold2=100) # Canny Edge Detection
    added_image = cv2.addWeighted(img_gray,1.0,edges,0.3,0)
    return added_image
    
mean, std = 0,0
for idx in range(len(labels)):
    image = transforms(
                read_image(head + '/' + labels.loc[idx]['Path'])
            ).float()
    # print(labels.loc[idx]['Path'])
    save_image((image / 255).float(), f'/central/groups/CS156b/2023/yasers_beavers/data/preprocessed/' + labels.loc[idx]['Path'].replace('/', '_'))
    
    sobelim = sobelfilter(f'/central/groups/CS156b/2023/yasers_beavers/data/preprocessed/' + labels.loc[idx]['Path'].replace('/', '_'))
    cv2.imwrite(f'/central/groups/CS156b/2023/yasers_beavers/data/sobel/overlay' + labels.loc[idx]['Path'].replace('/', '_'), sobelim)
    # save_image((image / 255).float(), f'/central/groups/CS156b/2023/yasers_beavers/data/sobel/' + labels.loc[idx]['Path'].replace('/', '_'))
    if idx % 10 == 9:
        print(f'{idx / len(labels) * 100}%')
    
    mean += image.mean()
    std += image.std()
    
print(f'Mean = {mean / len(labels)}') # used to be / 100
print(f'Std = {std / len(labels)}') # used to be / 100



