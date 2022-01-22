import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_blobs
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class DomianDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


def generate_data_loaders(X_train, X_test, Y_train, Y_test, batch_size=128):
    train_dataset = DomianDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_dataset = DomianDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader


def generate_toy_data(centers, stds, num_samples, num_features, train_percentage=80):
    train_num_sampels = [int(n*(train_percentage/100)) for n in num_samples]
    X_train, Y_train = make_blobs(n_samples=train_num_sampels,
                                  n_features=num_features,
                                  centers=centers,
                                  cluster_std=stds,
                                  random_state=0)
    test_num_sampels = [int(n * ((100-train_percentage) / 100)) for n in num_samples]
    X_test, Y_test = make_blobs(n_samples=test_num_sampels,
                                n_features=num_features,
                                centers=centers,
                                cluster_std=stds,
                                random_state=0)

    return X_train, X_test, Y_train, Y_test


def get_MNIST_data(i, batch_size=128):
    train_dataset = datasets.MNIST('data_MNIST_{}'.format(i), train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data_MNIST_{}'.format(i), train=False, transform=transforms.ToTensor())

    # a = np.random.choice(np.arange(len(train_dataset)), 1000, replace=False)
    # data = train_dataset.data.numpy()
    # labels = train_dataset.train_labels.numpy()
    # train_dataset = DomianDataset(data[a, None, :, :], labels[a])
    #
    # b = np.random.choice(np.arange(len(test_dataset)), 200, replace=False)
    # data = test_dataset.data.numpy()
    # labels = test_dataset.test_labels.numpy()
    # test_dataset = DomianDataset(data[b, None, :, :], labels[b])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader


def get_SVHN_data(batch_size=128):
    train_dataset = datasets.SVHN('data_SVHN', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.SVHN('data_SVHN', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return train_loader, test_loader


def get_USPS_data(batch_size=128):
    train_dataset = datasets.USPS('data_USPS', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.USPS('data_USPS', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return train_loader, test_loader



def plot_toy_data(X, Y, name):
    colors = ['red', 'green', 'blue', 'yellow', 'pink']
    plt.figure(figsize=(10, 10))
    plt.title("Data")

    x, y, C = X[:, 0], X[:, 1], np.array([colors[Y[i]] for i in range(len(Y))])
    plt.scatter(x, y, marker='o', c=C, s=30, edgecolor='k')
    plt.savefig(name+'.png')