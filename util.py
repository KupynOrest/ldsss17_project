import copy
import numpy as np
import os

import pickle

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time

from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print([round(cm[i][i], 2) for i in range(0, num_classes)])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))
    #     lr = init_lr #* 1 / (1 + )
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def copute_dataset_size(path=''):
    data_classes = [i for i in os.listdir(path) if not i.startswith('.')]
    num_entries = 0
    
    for folder in data_classes:
        current_dir = path + '/' + folder + '/'
        num_entries += len(os.listdir(current_dir))
    
    return num_entries

train_size = copute_dataset_size(path=data_dir + '/' + train_np_dir)
test_size = copute_dataset_size(path=data_dir + '/' + test_np_dir)

def check_size(address, sequence_size=50):
    x = np.load(address)
    print (x.shape)
    if x.shape[0] < sequence_size:
        return np.concatenate((x, np.zeros((sequence_size - x.shape[0], x.shape[1]))), axis=0)
    return x

# Data Loader (Input Pipeline)
def get_loader(path='', batch_size=batch_size):
    """
        Function reads .npy files from path.
        Returns:
        dataloader, data classes (list), size of input object [n_sequence, n_features], lenght_of_dataset
        """
    inputs = []
    targets = []
    data_classes = [i for i in os.listdir(path) if not i.startswith('.')]
    
    for folder in data_classes:
        current_dir = path + '/' + folder + '/'
        files = os.listdir(current_dir)
        #test_f = np.load(current_dir + files[0])[:,:30,:]
        
        #         print(test_f.shape)
        temp = [torch.Tensor(np.load(current_dir +  f).reshape((sequence_length, input_size))) for f in files]
        # Transform to torch tensors
        
        targets += ([torch.LongTensor([classes_dict[folder]])] * len(temp))
        inputs += temp
    
    tensor_x = torch.stack(inputs)
    tensor_y = torch.stack(targets)
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # Create your datset
    my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)  # Create your dataloader

    return (my_dataloader, data_classes)
