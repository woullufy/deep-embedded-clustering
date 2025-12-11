from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_fashion_mnist():
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return training_data, test_data


def load_mnist():
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return training_data, test_data


def create_dataloaders(train_data, test_data, batch_size=64, shuffle=True):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_input_matrix(dataset, device="cpu"):
    X = dataset.data.float().reshape(len(dataset), -1) / 255.0
    y = dataset.targets

    return X.to(device), y.to(device)
