import torch
from torch import nn
from torch.autograd import Variable

from bnlstm import LSTM, BNLSTMCell


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


class ConvLSTMBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(ConvLSTMBatchNorm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(cell_class=BNLSTMCell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, max_length=input_size).cuda()
        self.fc = nn.Linear(hidden_size, num_classes).cuda()
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        
        _, (out, _) = self.lstm(input_=x, hx=(h0, c0))
                                  
        # Decode hidden state of last time step
        out = self.fc(out[0])
        return out
