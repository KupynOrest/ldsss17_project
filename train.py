import datetime as dt
import os
import copy
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.utils.data

from torch.autograd import Variable
from models import ConvLSTM

from util import get_loader, exp_lr_scheduler, compute_dataset_size
from sklearn.metrics import confusion_matrix


def train_model(model, criterion, optimizer, lr_scheduler, loaders, **opts):
    since = time.time()
    
    best_model = model
    best_acc = 0.0
    
    for epoch in range(opts['num_epochs']):
        print('Epoch {}/{}'.format(epoch, opts['num_epochs'] - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in [opts['train_np_dir'], opts['test_np_dir']]:
            if phase == opts['train_np_dir']:
                optimizer = lr_scheduler(optimizer, epoch, init_lr=opts['learning_rate'])
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            dset_sizes = {
                opts['train_np_dir']: compute_dataset_size(path=os.path.join(opts['data_dir'], opts['train_np_dir'])),
                opts['test_np_dir']: compute_dataset_size(path=os.path.join(opts['data_dir'], opts['test_np_dir']))
            }

            # Iterate over data.
            for data in loaders[phase]:
                # get the inputs
                
                inputs, labels = data
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.view(-1, opts['sequence_length'], opts['input_size']).cuda())
                    labels = Variable(labels.view(-1).cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                #                 print (labels, outputs)
                
                # backward + optimize only if in training phase
                if phase == opts['train_np_dir']:
                    loss.backward()
                    optimizer.step()
                
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('running corrects: ', running_corrects)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == opts['test_np_dir'] and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                # saving weights
                torch.save(model, opts['data_dir'] + "/model_" + str(dt.time()) + str(epoch) + ".pt")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


default_opts = {
    'hidden_size': 256,
    'input_size': 512,
    'num_layers': 2,
    'dropout': 0.8,
    'batch_size': 2,
    'num_epochs': 200,
    'learning_rate': 0.1,
    'sequence_length': 50,
    'data_dir': 'data',
    'train_np_dir': 'train_np',
    'test_np_dir': 'test_np'
}

use_gpu = torch.cuda.is_available()
print("GPU is available: ", use_gpu)


def run(model_cls=ConvLSTM, opts=None):
    opts = opts or {}
    for k, v in default_opts.items():
        opts.setdefault(k, v)

    with open('classes_dict.pickle', 'rb') as f:
        classes = pickle.load(f)

    model = model_cls(input_size=opts['input_size'], hidden_size=opts['hidden_size'],
                      num_layers=opts['num_layers'], num_classes=len(classes),
                      dropout=opts['dropout'])
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts['learning_rate'], weight_decay=0)

    dset_loaders = {x: get_loader(opts['data_dir'] + '/' + x, classes,
                                  opts['sequence_length'], opts['input_size'],
                                  opts['batch_size'])[0] for x in [opts['train_np_dir'], opts['test_np_dir']]}

    model_lstm = train_model(model, criterion, optimizer, exp_lr_scheduler, dset_loaders, model_name='lstm', **opts)

    for data in get_loader(os.path.join(opts['data_np_dir'], opts['test_np_dir']), batch_size=3000)[0]:
        # get the inputs

        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            x_test = Variable(inputs.view(-1, opts['sequence_length'], opts['input_size']).cuda())
            y_test = labels.view(-1).cpu().numpy()
            y_pred = model_lstm(x_test)
            _, y_pred = torch.max(y_pred.data, 1)
            y_pred = y_pred.cpu().view(-1).numpy()
        break

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

#
# # Plot non-normalized confusion matrix
# # plt.figure()
# class_int_to_name = {classes_dict[key]:key for key in classes_dict}
# names_of_classes = [class_int_to_name[i] for i in range(0, num_classes)]
# print(len(names_of_classes), len(y_test))
# # plot_confusion_matrix(cnf_matrix, classes=names_of_classes,
# #                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# # plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=names_of_classes, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()
