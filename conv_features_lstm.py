import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from features.extractor import get_features

# Hyper Parameters
hidden_size = 128
input_size = 512
num_layers = 2
batch_size = 2
num_epochs = 2
learning_rate = 0.01
num_classes = 3
label_str_to_int = {'ApplyEyeMakeup': 0, 'Archery': 1, 'ApplyLipstick': 2}


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        print('x', type(x))
        # print (type(h0), type(c0))
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

rnn = RNN(input_size=512, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

sequence_length = 10

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (labels, images) in enumerate(
            get_features(in_dir='data/data_subset/', batch_size=1, max_frames=sequence_length)):
        print(images.size())
        labels = torch.LongTensor([label_str_to_int[i] for i in labels]).cuda()

        print('images before', type(images), type(labels))
        #         print (images)
        #         print(labels)

        images = images.view(-1, sequence_length, input_size)
        labels = labels.view(-1)

        print('images', type(images), type(labels))

        # Forward + Backward + Optimize

        optimizer.zero_grad()

        outputs = rnn(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
            % (epoch + 1, num_epochs, i + 1, 10 // batch_size, loss.data[0]))
