import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from utils import *

VIS_HEAD = '/groups/CS156b/2023/yasers_beavers/visualizations'

class ErrorPlotter():
    def __init__(self):
        self.losses = {'train': [], 'valid': []}
        self.accuracies = {'train': [], 'valid': [], 'batch' : []}

    def start_epoch(self):
        # train_loss, valid_loss, train_acc, valid_acc, train_total, valid_total
        self.epoch = [0] * 7 
        
    def update_train(self, loss, outputs, labels):
        self.epoch[0] += loss.item()
        self.epoch[2] += ((outputs > 0.5) == (labels > 0.5)).sum().item() / len(outputs)
        self.epoch[4] += 1
        self.epoch[6] = len(outputs)
        # print(len(outputs))

    def update_valid(self, criterion, outputs, labels):
        self.epoch[1] += criterion(outputs, labels.float()).item()
        self.epoch[3] += ((outputs > 0.5) == (labels > 0.5)).sum().item() / len(outputs)
        self.epoch[5] += 1 # len(labels)
        # self.epoch[6] = len(labels)

    def finish_epoch(self, epoch):
        self.losses['train'].append(self.epoch[0] / self.epoch[4])
        self.losses['valid'].append(self.epoch[1] / self.epoch[5])
        self.accuracies['train'].append(self.epoch[2] / self.epoch[4])
        self.accuracies['valid'].append(self.epoch[3] / self.epoch[5])
        self.accuracies['batch'].append(self.epoch[6])

        print_out(f'EPOCH {epoch:03}\t'\
                  f"Train Acc = {self.accuracies['train'][-1]:.6f}\t"\
                  f"Train Loss = {self.losses['train'][-1]:.6f}\t"\
                  f"Valid Acc = {self.accuracies['valid'][-1]:.6f}\t"\
                  f"Valid Loss = {self.losses['valid'][-1]:.6f}\t"\
                  f"Batch Size = {self.accuracies['batch'][-1]:03}\t"
        )

    def get_train(self):
        return self.accuracies['train'][-1]
    
    def get_valid(self):
        return self.accuracies['valid'][-1]
    
    def plot(self, vis_path, plot_title):
        for metric, lbl in zip(
            [self.losses, self.accuracies], ['loss', 'accuracy']
        ):
            plt.clf()
            for d in ['train', 'valid']:
                plt.plot(metric[d], label=d)
            plt.xlabel('epochs')
            plt.ylabel(lbl)
            plt.title(plot_title)
            plt.legend()
            plt.savefig(f'{vis_path}.png')

            with open(f'{vis_path}.pkl', 'wb') as f:
                pickle.dump(
                    {'losses': self.losses, 'accuracies': self.accuracies},
                    f
                )

class MultiErrorPlotter(ErrorPlotter):
    def update_valid(self, criterion, outputs, labels):
        self.epoch[1] += criterion(outputs, labels.float()).item()
        self.epoch[3] += ((outputs > 0.5) == (labels > 0.5))[0].sum().item()
        self.epoch[5] += len(labels)
