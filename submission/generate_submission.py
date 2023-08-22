import pandas as pd
import numpy as np

SUBMISSION_NAME = 'resnet.csv'
CLASSES = ['No Finding',
           'Enlarged Cardiomediastinum',
           'Cardiomegaly',
           'Lung Opacity',
           'Pneumonia',
           'Pleural Effusion',
           'Pleural Other',
           'Fracture',
           'Support Devices']
TEST_ID_PATH = r'/central/groups/CS156b/2023/yasers_beavers/submission/test2023_path_to_id.csv'
TRAIN_ID_PATH = r'/central/groups/CS156b/2023/yasers_beavers/data/train2023.csv'

test_df = pd.read_csv(TEST_ID_PATH)
train_df = pd.read_csv(TRAIN_ID_PATH)
df = pd.DataFrame()
filename_to_class = {r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Pneumonia_epochs_15_train_1.0.csv' : ['Pneumonia'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Enlarged_Cardiomediastinum_epochs_15_train_1.0.csv' : ['Enlarged Cardiomediastinum'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Fracture_epochs_15_train_1.0.csv': ['Fracture'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Pleural_Effusion_epochs_15_train_0.5.csv' : ['Pleural Effusion'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Cardiomegaly_epochs_15_train_1.0.csv' : ['Cardiomegaly'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Pleural_Other_epochs_15_train_1.0.csv' : ['Pleural Other'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Support_Devices_epochs_15_train_1.0.csv' : ['Support Devices'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_No_Finding_epochs_15_train_0.5.csv' : ['No Finding'],
                     r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet101_Lung_Opacity_epochs_15_train_1.0.csv' : ['Lung Opacity']}
# filename_to_class = {}
# r'/central/groups/CS156b/2023/yasers_beavers/predictions/resnet18_Support_Devices_epochs_30_train_1.0.csv' : ['Support Devices']
def float_to_class(x):
    if x < 0.3:
        x = 0
    elif x > 0.7:
        x = 1
        
    return x * 2 - 1

for file in filename_to_class:
    col = filename_to_class[file][0]
    df[col] = pd.read_csv(file)[col]
    # df = df.drop('Unnamed: 0', axis=1)
    # print(pd.read_csv(file))

for col in df:
    df[col] = df[col].apply(float_to_class)
    print(f'{col}: Mean = {df[col].mean()} Std = {df[col].std()}')

    print(len(df), len(test_df))
    assert len(df) == len(test_df)

for row in CLASSES:
    if row not in df:
        # df[row] = 2 * np.random.random(len(test_df)) - 1
        mu = train_df[row].mean()
        df[row] = mu
        print('Averaged for', row, '=', mu)

for row in pd.read_csv(TEST_ID_PATH):
    df[row] = test_df[row]

df.to_csv(SUBMISSION_NAME)