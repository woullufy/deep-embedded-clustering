import torch

from torch.utils.data import DataLoader

from models import Autoencoder, DEC

from utils import *
from utils.training_dec import train_dec

def main():
    training_data, testing_data = load_fashion_mnist()

    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=64, shuffle=True)

    ae = Autoencoder()
    ae.load_state_dict(torch.load('checkpoints/autoencoder_param.pth'))

    dec = DEC(ae, 10)
    optimizer = torch.optim.Adam(dec.parameters(), lr=1e-3)
    dec_losses = train_dec(dec, train_loader, optimizer, epochs=10, device='cpu')


if __name__ == '__main__':
    main()
