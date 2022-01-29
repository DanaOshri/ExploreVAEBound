import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.datasets import make_blobs
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
import os

class MyDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


def generate_data_loaders(X_train, X_test, Y_train, Y_test, batch_size=128):
    train_dataset = MyDataset(X_train[:, None, :, :], Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_dataset = MyDataset(X_test[:, None, :, :], Y_test)
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
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test


# scale pixels
def prep_pixels(images):
    # convert from integers to floats
    images_norm = images.astype('float32')

    # normalize to range 0-1
    images_norm = images_norm / 255.0

    # return normalized images
    return images_norm


def get_USPS_data(name="USPS", batch_size=128):

    train_dataset = datasets.USPS('data_USPS', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.USPS('data_USPS', train=False, transform=transforms.ToTensor())

    all_images = np.concatenate((train_dataset.data, test_dataset.data))
    all_labels = np.concatenate((train_dataset.targets, test_dataset.targets))

    indices = np.arange(len(all_images))
    sample_indices = np.random.choice(indices, 1000, replace=False)
    all_images = all_images[sample_indices]
    all_labels = all_labels[sample_indices]

    plt.imshow(all_images[0], cmap='gray')

    all_images = prep_pixels(all_images)

    plt.imshow(all_images[0], cmap='gray', vmin=0, vmax=1)

    X_train, X_test, y_train, y_test = train_test_split(all_images,
                                                        all_labels,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=all_labels)

    train_loader, test_loader = generate_data_loaders(X_train, X_test, y_train, y_test, batch_size)
    return train_loader, test_loader


def get_MNIST_data(batch_size=128):
    train_dataset = datasets.MNIST('data_MNIST', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data_MNIST', train=False, transform=transforms.ToTensor())

    all_images = np.concatenate((train_dataset.data.numpy(), test_dataset.data.numpy()))
    all_labels = np.concatenate((train_dataset.targets.numpy(), test_dataset.targets.numpy()))

    indices = np.arange(len(all_images))
    sample_indices = np.random.choice(indices, 1000, replace=False)
    all_images = all_images[sample_indices]
    all_labels = all_labels[sample_indices]

    all_images = prep_pixels(all_images)

    X_train, X_test, y_train, y_test = train_test_split(all_images,
                                                        all_labels,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=all_labels)

    train_loader, test_loader = generate_data_loaders(X_train, X_test, y_train, y_test, batch_size)
    return train_loader, test_loader


def get_MNIST_16_data(batch_size=128):
    train_dataset = datasets.MNIST('data_MNIST', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data_MNIST', train=False, transform=transforms.ToTensor())

    all_images = np.concatenate((train_dataset.data.numpy(), test_dataset.data.numpy()))
    all_labels = np.concatenate((train_dataset.targets.numpy(), test_dataset.targets.numpy()))

    indices = np.arange(len(all_images))
    sample_indices = np.random.choice(indices, 1000, replace=False)
    all_images = all_images[sample_indices]
    all_labels = all_labels[sample_indices]

    # plt.imshow(all_images[0], cmap='gray', vmin=0, vmax=255)

    all_images = prep_pixels(all_images)
    all_images = np.array([resize(img, (16, 16)) for img in all_images])

    # plt.imshow(all_images[0], cmap='gray', vmin=0, vmax=1)

    X_train, X_test, y_train, y_test = train_test_split(all_images,
                                                        all_labels,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=all_labels)

    train_loader, test_loader = generate_data_loaders(X_train, X_test, y_train, y_test, batch_size)
    return train_loader, test_loader


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_SVHN_data(batch_size=128):
    svhn_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.SVHN('data_SVHN', split='train', download=True, transform=svhn_transform)
    test_dataset = datasets.SVHN('data_SVHN', split='test', download=True, transform=svhn_transform)

    all_images = np.concatenate((train_dataset.data, test_dataset.data))
    all_labels = np.concatenate((train_dataset.labels, test_dataset.labels))

    indices = np.arange(len(all_images))
    sample_indices = np.random.choice(indices, 1000, replace=False)
    all_images = all_images[sample_indices]
    all_labels = all_labels[sample_indices]
    #
    # img = np.transpose(all_images[0], [1, 2, 0])
    # plt.subplot(121)
    # plt.imshow(np.squeeze(img))
    img = torch.tensor(all_images[0])
    plt.imshow(img.permute(1, 2, 0).numpy())

    all_images = [rgb2gray(img) for img in all_images]
    all_images = np.array([resize(img, (16, 16)) for img in all_images])
    all_images = prep_pixels(all_images)

    plt.imshow(all_images[0], cmap='gray', vmin=0, vmax=1)


    X_train, X_test, y_train, y_test = train_test_split(all_images,
                                                        all_labels,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=all_labels)

    train_loader, test_loader = generate_data_loaders(X_train, X_test, y_train, y_test, batch_size)
    return train_loader, test_loader



def plot_toy_data(X, Y, name):
    colors = ['red', 'green', 'blue', 'yellow', 'pink']
    plt.figure(figsize=(10, 10))
    plt.title("Data")

    x, y, C = X[:, 0], X[:, 1], np.array([colors[Y[i]] for i in range(len(Y))])
    plt.scatter(x, y, marker='o', c=C, s=30, edgecolor='k')
    plt.savefig(name+'.png')