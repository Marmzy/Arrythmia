#!/usr/bin/env python

import argparse
import numpy as np
import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from CustomECGDataset import ECGDataset
from torch.utils.data import DataLoader
from torchvision import models


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('--data', type=str, help='Path to the data directory')

    #Options for the model
    parser.add_argument('--layers', type=str, default="34", help='Number of layers for the ResNet model (18, default=34, 50, 101, 152)')
    parser.add_argument('--pretrained', type=bool, default=True, nargs='?', help='Indicates whether the ResNet model is pretrained on ImageNet or not (default: True)')

    #Options for the optimizer
    parser.add_argument('--lr', type=float, help='ADAM gradient descent optimizer learning rate')
    parser.add_argument('--decay', type=float, default=0.0, nargs='?', help='ADAM weight decay (default: 0.0)')

    #Options for the loss function
    parser.add_argument('--weight', type=float, default=1.0, nargs='?', help='Weight of the Normal heartbeat class for the loss function (default: 1.0)')

    #Options for training
    parser.add_argument('--epochs', type=int, help='Number of epochs')

    #Printing arguments to the command line
    args = parser.parse_args()
    print('Called with args:')
    print(args)

    return args


def check_data(data, path):

    #Loading the data
    X_train = np.load(os.path.join(path, data, "train", "X_train.npy"))
    y_train = np.load(os.path.join(path, data, "train", "y_train.npy"))

    X_test = np.load(os.path.join(path, data, "test", "X_test.npy"))
    y_test = np.load(os.path.join(path, data, "test", "y_test.npy"))

    #Checking the input shape of the data
#    print("\nInput training data shape:\n{}".format(X_train.shape))
#    print("Input test data shape:\n{}\n".format(X_test.shape))

    #Checking for class imbalance
    label0 = y_train.tolist().count(0)
    label1 = y_train.tolist().count(1)

    print("Checking for class imbalance...\n")
    print("Number of 0 labels: {}".format(label0))
    print("Number of 1 labels: {}".format(label1))

    if label0 / label1 > 2 or label0 / label1 < 0.5:
        print("\t=> The dataset is quite imbalanced: ratio ~= {:.2f}\n".format(label0 / label1))


def load_data(data, path, train):

    #Constructing the PyTorch DataSet
    if train:
        train_data = ECGDataset(os.path.join(path, data, "train", "X_train.npy"), os.path.join(path, data, "train", "y_train.npy"))
        return DataLoader(train_data, batch_size=64, shuffle=True)
    else:
        test_data = ECGDataset(os.path.join(path, data, "test", "X_test.npy"), os.path.join(path, data, "test", "y_test.npy"))
        return DataLoader(test_data, batch_size=64, shuffle=True)


def train_model(model, loss, optimizer, epochs, data_loader, fout):

    #Initialising variables
    pkl_queue = deque()
    best_acc = -1.0
    best_sens = -1.0
    best_loss = 100.0
    best_model_weights = model.state_dict()

    print(model, "\n")

    #Looping over the epochs
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch, epochs))

        #Making sure training and validating the model is different
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            #Setting the data
            data = data_loader[phase]

            #Initialising more variables
            running_loss = 0.0
            normal_correct = 0
            ventricular_correct = 0

            #Looping over the minibatches
            for idx, (data_train, target_train) in enumerate(data):
                optimizer.zero_grad()
                x, y = data_train.to(device), target_train.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)



def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))

    #Checking the data
    check_data(args.data, path)

    #Loading the data
    train_data = load_data(args.data, path, train=True)
    val_data = load_data(args.data, path, train=False)
    data_loader = {"train": train_data,
                   "val": val_data}

    #Initialising the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.pretrained:
        model = models.__dict__["resnet"+args.layers](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2, bias=True)
    else:
        model = models.__dict__["resnet"+args.layers]
        model.fc = nn.Linear(model.fc.in_features, 2, bias=True)


    #Settings for training the model
    model.to(device)
    fout = os.path.join("/".join(os.getcwd().split("/")[:-1]), "data", "train")
    epochs = args.epochs
    weight = torch.FloatTensor([args.weight, 1.])
    loss = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    #Training the model
    train_model(model, loss, optimizer, epochs, data_loader, fout)


if __name__ == '__main__':
    main()
