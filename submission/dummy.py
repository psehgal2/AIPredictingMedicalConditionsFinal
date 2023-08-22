import pandas as pd
import numpy as np

SAVE_FILE = r'random.csv'
TEST_ID_PATH = r'test2023_path_to_id.csv'

df = pd.read_csv(TEST_ID_PATH)

df.drop('Path', axis=1)

for row in 'No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Pneumonia,Pleural Effusion,Pleural Other,Fracture,Support Devices'.split(','):
    df[row] = np.round(2 * np.random.random(len(df)) - 1).astype(int)

df.to_csv(SAVE_FILE)
