import torch
import torch.nn as nn

class DEC(nn.Module):
    def __init__(self, autoencoder):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder


    def forward(self, x):
        pass