import numpy as np
import torch
import torchvision
import pandas as pd
import csv
import os


def tanh(a):
    vec_tanh = np.vectorize(lambda x: np.tanh(x))
    return vec_tanh(a)

def dev_tanh(a):
    vec_dev_tanh = np.vectorize(lambda x: 1 / np.cosh(x) ** 2)
    return vec_dev_tanh(a)

def dev2_tanh(a):
    vec_dev2_tanh = np.vectorize(lambda x: - 2 * np.tanh(x) * (1 - np.tanh(x)))
    return vec_dev2_tanh(a)


def square(a):
    vec_square = np.vectorize(lambda x: x ** 2 / 2)
    return vec_square(a)

def dev_square(a):
    vec_dev_square = np.vectorize(lambda x: x)
    return vec_dev_square(a)

def dev2_square(a):
    vec_dev2_square = np.vectorize(lambda x: 1)
    return vec_dev2_square(a)


def cos(a):
    vec_cos = np.vectorize(lambda x: np.cos(x))
    return vec_cos(a)

def dev_cos(a):
    vec_dev_cos = np.vectorize(lambda x: - np.sin(x))
    return vec_dev_cos(a)

def dev2_cos(a):
    vec_dev2_cos = np.vectorize(lambda x: - np.cos(x))
    return vec_dev2_cos(a)


def softplus(a):
    vec_cos = np.vectorize(lambda x: np.log(1 + np.exp(x)))
    return vec_cos(a)

def dev_softplus(a):
    vec_dev_cos = np.vectorize(lambda x: np.exp(x) / (np.exp(x) + 1))
    return vec_dev_cos(a)

def dev2_softplus(a):
    vec_dev2_cos = np.vectorize(lambda x: np.exp(x) / (np.exp(x) + 1) ** 2)
    return vec_dev2_cos(a)


def rwt(a, b):
    out = []
    if len(a) != len(b):
        print('Error')
        return None
    for i in range(len(a)):
        out.append(np.kron(a[i], b[i]))
    return np.array(out)




def load_dataset(dataset, N, class1, class2):

    if dataset == 'MNIST':
        dataset_train = torchvision.datasets.MNIST('./data/', train=True, download=True)
        dataset_test = torchvision.datasets.MNIST('./data/', train=False, download=True)
    if dataset == 'CIFAR-10':
        dataset_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
        dataset_test = torchvision.datasets.CIFAR10('./data', train=False, download=True)

    dataset_train.targets = np.array(dataset_train.targets)
    dataset_test.targets = np.array(dataset_test.targets)

    indices_train = (dataset_train.targets == class1) | (dataset_train.targets == class2)
    dataset_train.data, dataset_train.targets = dataset_train.data[indices_train], dataset_train.targets[indices_train]
    dataset_train.targets = 1 * (dataset_train.targets == class1) - 1 * (dataset_train.targets == class2)

    indices = np.random.choice(range(10000), size=N, replace=False, p=None)

    dataset_train.data = dataset_train.data[indices]
    dataset_train.targets = dataset_train.targets[indices]

    indices_test = (dataset_test.targets == class1) | (dataset_test.targets == class2)
    dataset_test.data, dataset_test.targets = dataset_test.data[indices_test], dataset_test.targets[indices_test]
    dataset_test.targets = 1 * (dataset_test.targets == class1) - 1 * (dataset_test.targets == class2)

    dataset_train_flat = dataset_train.data.reshape(dataset_train.data.shape[0], -1)
    dataset_test_flat = dataset_test.data.reshape(dataset_test.data.shape[0], -1)

    dataset_train_flat = np.array(dataset_train_flat / 255)
    dataset_test_flat = np.array(dataset_test_flat / 255)

    targets_test = np.array(dataset_test.targets)
    targets_train = np.array(dataset_train.targets)
    
    return dataset_train_flat, dataset_test_flat[0], targets_train, targets_test


def import_in_df(folder, activation):

    data = []
    my_folder = folder + activation

    for filename in os.listdir(my_folder):
        if '.txt' in filename:
            with open(os.path.join(my_folder, filename), 'r') as f:
                reader = csv.reader(f,  delimiter='\t')
                for row in reader:
                    new_row = []
                    for j in range(3):
                        new_row.append(int(row[j]))
                    for j in range(3, 4):
                        new_row.append(float(row[j]))
                    data.append(new_row)

    df = pd.DataFrame(data=data, columns=(['d', 'k', 'N', 'S']))
    df['activation'] = activation
    
    return df