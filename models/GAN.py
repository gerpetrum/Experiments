import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

count_class = 10

class FullyConnected(nn.Module):
    def __init__(self, sizes, dropout=False, activation_fn=nn.Tanh(), flatten=False, last_fn=None):
        super(FullyConnected, self).__init__()
        layers = []
        self.flatten = flatten
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(activation_fn) # нам не нужен дропаут и фнкция активации в последнем слое
        else: 
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if last_fn is not None:
            layers.append(last_fn)
        self.model = nn.Sequential(*layers)
    
    def label2vec(self, batch):
        b = np.zeros((len(batch), count_class))
        b[np.arange(len(batch)), torch.Tensor.numpy(batch)] = 1
        return torch.Tensor(b)
    
    def forward(self, x, y):
        if self.flatten:
            x = x.view(x.shape[0], -1)
        return self.model(torch.cat([x, self.label2vec(y)], 1))
