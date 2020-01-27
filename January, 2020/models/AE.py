import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, latent_size=10):
        super(Encoder, self).__init__() # функция создаёт объект базового класса
        self.fc1 = nn.Linear(28*28, latent_size)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x
    

class Decoder(nn.Module):
    def __init__(self, latent_size=10):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 28*28)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x
    

class AE(nn.Module):
    def __init__(self, latent_size=10, loss_fn=F.mse_loss, lr=1e-4, l2=0.):
        super(AE, self).__init__()
        self.latent_size = latent_size
        self.E = Encoder(latent_size)
        self.D = Decoder(latent_size)
        self.loss_fn = loss_fn
        self._rho_loss = None
        
        self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        
    def forward(self, x):
        x = x.view(-1, 28*28) # reshape tensor to tensor with 28*28 columns
        h = self.E(x)
        self.data_rho = h.abs().sum(0)
        out = self.D(h)
        return out

    def encode(self, h):
        with torch.no_grad():
            return self.E(h)
        
    def decode(self, h):
        with torch.no_grad():
            return self.D(h)
    
    # функция после декодера
    def rho_loss(self, rho, size_average=True):
        """
        D_KL(P||Q) = sum(p*log(p/q)) = -sum(p*log(q/p)) = -p*log(q/p) - (1-p)log((1-q)/(1-p))
        """
        dkl = -torch.log(self.data_rho) * rho - torch.log(1 - self.data_rho) * (1 - rho)
        if size_average:
            self._rho_loss = dkl.mean()
        else:
            self._rho_loss = dkl.sum()
        return self._rho_loss
        
    def loss(self, x, target, **kwargs):
        target = target.view(-1, 28*28)
        self._loss = self.loss_fn(x, target, **kwargs)
        return self._loss