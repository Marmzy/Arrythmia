#!/usr/bin/env python

import argparse
import copy
import numpy as np
import os
import pickle as pkl
import time
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from CNN import ResNet34
from CustomECGDataset import ECGDataset
from torch.utils.data import DataLoader
from torchvision import models


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('--data', type=str, help='Path to the data directory')

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


def train_model(model, loss, optimizer, epochs, data_loader, fout, device):

    #Initialising variables
    pkl_queue = deque()
    best_acc = -1.0
    best_sens = -1.0
    best_loss = 100.0
    best_model_weights = model.state_dict()
    end = time.time()

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
            running_loss = 0
            running_correct = 0
            ventricular_correct = 0
            ventricular_size = 0

            #Looping over the minibatches
            for idx, (data_train, target_train) in enumerate(data):
                optimizer.zero_grad()
                x, y = data_train.to(device), target_train.to(device)
                print(x)
                print(x[0])
                break           ##################################################

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)
                    _, predictions = torch.max(y_pred.data, 1)
                    l = loss(y_pred, y)

                    if phase == "train":
                        l.backward()
                        optimizer.step()

                #Calculating statistics
                running_loss += l.item() * x.size(0)
                running_correct += torch.eq(predictions, y).sum()
                ventricular_size += torch.sum(y)
                ventricular_correct += torch.eq(y + predictions, torch.tensor([2]*len(y)).to(device)).sum().to(device)

            break #######################################

            #Calculating mean statistics
            epoch_loss = running_loss / len(x)
            epoch_acc = running_correct.double() / len(x)
            epoch_sens = 0.0
            if ventricular_size > 0.0:
                epoch_sens = float(ventricular_correct) / float(ventricular_size)

            print("\t{} Loss: {:.4f} Acc: {:.4f} Sen: {:.4f} Time: {:.4f}".format(phase, epoch_loss, epoch_acc, epoch_sens, time.time()-end), end="")


            #Saving the best acc/sens model for the validation data
            if phase == "val":
                if (epoch_acc > best_acc) or (epoch_acc == best_acc and epoch_sens > best_sens):
                    best_sens = epoch_sens
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
#                    torch.save(model.state_dict(), "{}_epoch{}.pkl".format(fout, epoch))



def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    fdir = os.path.join("/".join(os.getcwd().split("/")[:-1]), "data")
    fout = os.path.join(fdir, "ResNet34_w{}_lr{}_decay{}".format(args.weight, args.lr, args.decay))

    #Checking the data
    check_data(args.data, path)

    #Loading the data
    train_data = load_data(args.data, path, train=True)
    val_data = load_data(args.data, path, train=False)
    data_loader = {"train": train_data,
                   "val": val_data}

    #Initialising the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet34().to(device)

    #Settings for training the model
    weight = torch.FloatTensor([args.weight, 1.]).to(device)
    loss = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    #Training the model
    train_model(model, loss, optimizer, args.epochs, data_loader, fout, device)


if __name__ == '__main__':
    main()
