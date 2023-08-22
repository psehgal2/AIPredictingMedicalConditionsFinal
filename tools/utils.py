"""
File for general utility things...
"""
import sys
from datetime import datetime

import pickle
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIRECTORY = r'/central/groups/CS156b/data/'
TEAM_DIRECTORY = r'/central/groups/CS156b/2023/yasers_beavers'

def get_image_df(file='train2023.csv'):
    """
    Reads in data information and path for all training images. When path
    concatenated with DATA_DIRECTORY, the file can be accessed with python
    """
    df = pd.read_csv(DATA_DIRECTORY + '/student_labels/' + file)
    return df

def time_str():
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")

def print_out(*a):
    print(*a, file=sys.stdout)

def plot_loss(pkl, out):
    with open(pkl, 'rb') as f:
        f = pickle.load(f)

    for (n1, n2) in [['loss', 'losses'], ['accuracy', 'accuracies']]:
        plt.clf()
        for x in ['train', 'valid']:
            plt.plot(f[n2][x], label=x)
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(f'{out}_{n1}.png')