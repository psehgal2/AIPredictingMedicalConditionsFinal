import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip
from torchvision.io import read_image, ImageReadMode

head = "/groups/CS156b/data"

'''
TRAIN, VALID, TEST GENERAL DATASET
'''
class TestDataset(Dataset):
    def __init__(self, annotations, img_dir, transforms=[], cpu=False, size=250, crop=0.9):
        if cpu:
            self.labels = (
                pd.read_csv(annotations).sample(frac=0.01).reset_index(drop=True)
            )
            print('WARNING: To predict on 1% of test data')
        else:
            self.labels = pd.read_csv(annotations)
        print('Test set size =', len(self.labels))
        self.images = img_dir
        print('Image size =', size)
        self.transforms = torch.nn.Sequential(
            Resize(int(size / crop), antialias=True),
            RandomCrop(size),
            RandomHorizontalFlip()
        )
        self.extra_transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transforms(
            read_image(head + "/" + self.labels.loc[idx]["Path"], ImageReadMode.RGB)
        )
        return image.float() / 255.0

class TrainValidDatasets:
    def __init__(
        self,
        clf,
        annotations,
        img_dir,
        condition,
        downsample=1,
        train_split=0.8,
        transforms=[],
        drop=True,
        mc=False,
        size=250,
        crop=0.9
    ):
        print(condition)
        if drop:
            df = (
                pd.read_csv(annotations)
                .dropna(subset=condition)
                .sample(frac=downsample)
            )
        else:
            df = pd.read_csv(annotations).sample(frac=downsample)
        if drop:
            for c in condition:
                df = df[df[condition] != 0]
        if mc:
            for c in condition:
                df[c] = df[c].apply(
                    lambda x: 1 if x == 1 else 0
                )
        else:
            df[condition] = df[condition].apply(lambda x: 1 if x == 1 else 0)

        majority = 1 if len(df[df[condition] == 1]) / len(df) > 0.5 else 0
        larger = 0.5
        # Class balancing by pruning data points randomly
        while len(df[df[condition] == majority]) / len(df) > 0.5:
            row = df.sample(axis=0)
            if row[condition].item() == majority:
                df = df.drop(row.index.item())
                larger = len(df[df[condition] == 1]) / len(df)
        print(f'TrainValidDataset is {larger * 100}% 1s out of {len(df)} total points')
        
        is_train = np.random.rand(len(df)) < train_split
        self.train = clf( # BinaryPathologyDataset or MultiPathologyDataset
            df[is_train].reset_index(drop=True), img_dir, condition, transforms, mc=mc, size=size, crop=crop
        )
        self.valid = clf( # BinaryPathologyDataset or MultiPathologyDataset
            df[~is_train].reset_index(drop=True), img_dir, condition, transforms, mc=mc, size=size, crop=crop
        )

'''
BINARY PATHOLOGY DATASET CLASSES
'''
class BinaryPathologyDataset(Dataset):
    def __init__(self, label_df, img_dir, condition, transforms=[], mc=False, size=250, crop=0.9):
        self.labels = label_df
        self.images = img_dir
        self.condition = condition
        print('Image size =', size)
        self.transforms = torch.nn.Sequential(
            Resize(int(size / crop), antialias=True), 
            RandomCrop(size), 
            RandomHorizontalFlip()
        )
        self.extra_transforms = transforms
        self.mc = mc

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transforms(
            read_image(head + "/" + self.labels.loc[idx]["Path"], ImageReadMode.RGB)
        )
        # print(image.float().mean())
        label = self.labels.loc[idx][self.condition]
        if self.mc:
            label = torch.from_numpy(label.to_numpy(dtype=int))
        return image.float() / 255, int(label)

class BinaryPathologyDatasets(TrainValidDatasets):
    def __init__(
        self,
        annotations,
        img_dir,
        condition,
        downsample=1,
        train_split=0.8,
        transforms=[],
        drop=True,
        mc=False,
        size=250,
        crop=0.9
    ):
        TrainValidDatasets.__init__(
            self,
            BinaryPathologyDataset,
            annotations=annotations,
            img_dir=img_dir, 
            condition=condition,
            downsample=downsample,
            train_split=train_split,
            transforms=transforms,
            drop=drop,
            mc=mc,
            size=size,
            crop=crop
        )



'''
COMBINED MULTIPATHOLOGY DATASET CLASSES
'''
class MultiPathologyDataset(BinaryPathologyDataset):
    def __getitem__(self, idx):
        try:
            image = self.transforms(
                read_image(head + "/" + self.labels.loc[idx]["Path"], ImageReadMode.RGB)
            )
        except:
            print(idx, self.labels.loc[idx]["Path"], self.labels[:20])
        label = int(sum([self.labels.loc[idx][cond] for cond in self.condition]) > 0)
        return image, label

class MultiPathologyDatasets(TrainValidDatasets):
    def __init__(
        self,
        annotations,
        img_dir,
        condition,
        downsample=1,
        train_split=0.8,
        transforms=[],
        drop=True,
        mc=True,
        crop=0.9
    ):
        TrainValidDatasets.__init__(
            self,
            MultiPathologyDataset,
            annotations=annotations,
            img_dir=img_dir,
            condition=condition,
            downsample=downsample,
            train_split=train_split,
            transforms=transforms,
            drop=drop,
            mc=mc,
            crop=crop
        )

class HierarchyDatasets:
    def __init__(
        self,
        annotations,
        img_dir,
        conditions,
        downsample=1,
        train_split=0.8,
        transforms=[],
        drop=True,
    ):
        """
        conditions [['Pneumonia', 'Support Devices'], ['Fracture']]
        """
        # conditions = np.array(conditions).flatten().tolist()
        self.datasets_l1 = [
            MultiPathologyDatasets(
                annotations=annotations,
                img_dir=img_dir,
                condition=condition_set,
                downsample=downsample,
                train_split=train_split,
                transforms=transforms,
                drop=drop,
                mc=True,
            )
            for condition_set in conditions
        ]

'''
RECONSTRUCTION DATASET
'''
class ReconstructionDataset(Dataset):
    def __init__(self, annotations, img_dir, transforms=[], size=256):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transforms = torch.nn.Sequential(
            Resize(size, antialias=True), 
            RandomCrop(int(1 * size)), 
            RandomHorizontalFlip()
        )
        self.extra_transforms = transforms

    def __len__(self): 
        return len(self.annotations)

    def __getitem__(self, idx):
        image = self.transforms(read_image(
            f"{self.img_dir}/{self.annotations.loc[idx]['Path']}", ImageReadMode.RGB
        ))
        features = self.annotations.loc[idx][
            ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
        ]
        return (image / 255).float()# , features

class ReconstructionDatasets():
    def __init__(
        self, 
        annotations, 
        img_dir, 
        downsample=1, 
        train_split=0.8, 
        transforms=[], 
        size=256,
        orientation=None,
    ):
        self.a = pd.read_csv(annotations).sample(frac=downsample)
        if orientation is not None:
            self.a = self.a[self.a['Frontal/Lateral'] == orientation]
        is_train = np.random.rand(len(self.a)) < train_split
        self.train = ReconstructionDataset(
            self.a[is_train].reset_index(drop=True), img_dir, transforms, size
        )
        self.valid = ReconstructionDataset(
            self.a[~is_train].reset_index(drop=True), img_dir, transforms, size
        )

 

    