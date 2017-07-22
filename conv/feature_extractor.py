from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os

data_transforms = {
    'images': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data_subset'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['images']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=1, shuffle=False, num_workers=8)
                for x in ['images']}
dset_sizes = {x: len(dsets[x]) for x in ['images']}
dset_classes = dsets['images'].classes

use_gpu = torch.cuda.is_available()

# Get a batch of training data
inputs, classes = next(iter(dset_loaders['images']))

model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features
num_classes = 512
model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft.fc.weight.data.zero_()

if use_gpu:
    model_ft = model_ft.cuda()

def get_features(model, loader):
    model.train(False)  # Set model to evaluate mode

    output_list = []

    for data in loader:
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        numpy_out = outputs.data.numpy()[0] * 100

        output_list.append(numpy_out)
        print(len(output_list))

    return output_list

# Train and evaluate ^^^^^^^^^^^^^^^^^^
features = get_features(model_ft, dset_loaders['images'])


