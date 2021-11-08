#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn

from CNN import ResNet34
from CustomECGDataset import ECGDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def parseArgs():

    parser = argparse.ArgumentParser(description='Evaluating models on the test dataset')

    #Options for input and output
    parser.add_argument('--indir', type=str, help='Name of the data directory')
    parser.add_argument('--infiles', type=str, help='Stem name of the input files')
    parser.add_argument('--normalised', type=str2bool, nargs='?', const=True, default=False, help='Use the normalised dataset')
    parser.add_argument('--denoised', type=str2bool, nargs='?', const=True, default=False, help='Use the denoised dataset')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')

    #Options for class imbalance
    parser.add_argument('--sampling', type=str2bool, nargs='?', const=True, default=False, help='Randomly sample heartbeats from the majority class to match the number of the minority class')
    parser.add_argument('--weight', type=str2bool, nargs='?', const=True, default=False, help='Rescale the weights of the loss function to alleviate class imbalance')

    #Printing arguments to the command line
    args = parser.parse_args()

    #Checking arguments
    assert not (args.normalised and args.denoised), 'Arguments --normalised and --denoised are not compatible. Please choose one.'

    print('Called with args:')
    print(args)

    return args


def get_weights(path, normalise, denoise, verbose):

    #Initialising variable
    suffix = ""
    if normalise:
        suffix += "_normalised"
    elif denoise:
        suffix += "_denoised"

    #Loading the data
    y_test = np.load(os.path.join(path, "y_test.npy"))

    #Checking for class imbalance
    label0 = y_test.tolist().count(0)
    label1 = y_test.tolist().count(1)

    #Calculate weights to alleviate class imbalance
    if verbose:
        print("Getting the class weights...\n")
    weights = [1 - (samples / (label0 + label1)) for samples in [label0, label1]]
    return weights


def load_data(path, normalise, denoise):

    #Initialising variable
    suffix = ""
    if normalise:
        suffix += "_normalised"
    elif denoise:
        suffix += "_denoised"

    #Constructing the PyTorch DataSet
    test_data = ECGDataset(os.path.join(path, "X_test{}.npy".format(suffix)), os.path.join(path, "y_test.npy"))
    return DataLoader(test_data, batch_size=64, shuffle=True)


def make_preds(model, loss, data_loader, k, fout, device, verbose):

    #Writing the predictions to a file
    with open(fout, "a") as f:
        print("Evaluating the model trained on training dataset fold {}...".format(k))

        #Setting the data and model
        data = data_loader["test"]
        model.train(False)

        #Looping over the minibatches
        for idx, (data_test, target_test) in enumerate(data):
            x, y = data_test.to(device), target_test.to(device)

            with torch.set_grad_enabled(False):
                y_pred = model(x)
                prob = nn.functional.softmax(y_pred, dim=1)
                _, predictions = torch.max(y_pred.data, 1)
                l = loss(y_pred, y)

        #Saving the predictions
        for i in range(len((y_pred))):
            print("{}\t{}\t{:.5f}\t{:.5f}".format(predictions[i].item(), y[i].item(), prob[i][0], prob[i][1]), file=f)


def score(fout):

    #Loading the predictions data
    df = pd.read_csv(fout, sep="\t")
    truth = df.loc[:, "GroundTruth"].to_numpy()
    preds = df.loc[:, "Prediction"].to_numpy()

    #Calculating various metrics
    tn, fp, fn, tp = confusion_matrix(truth, preds).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = tp / (tp + (0.5 * (fp + fn)))
    auc_score = roc_auc_score(truth, preds)

    #Outputting the metrics
    with open(fout, "a") as f:
        print("\nModel evaluation metrics:")
        print("\nModel evaluation metrics:", file=f)
        print("Accuracy\tSensitivity\tSpecificity\tPrecision\tF1-Score\tAUC Score")
        print("Accuracy\tSensitivity\tSpecificity\tPrecision\tF1-Score\tAUC Score", file=f)
        print("{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}".format(accuracy, sensitivity, specificity, precision, f1_score, auc_score))
        print("{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}".format(accuracy, sensitivity, specificity, precision, f1_score, auc_score), file=f)


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    test_dir = os.path.join(path, args.indir, "test")
    fout = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_predictions.txt")

    #Writing the header to the output file
    with open(fout, 'w') as f:
        f.write("Prediction\tGroundTruth\tProbClass0\tProbClass1\n")

    #Loading the test dataset
    if args.verbose:
        print("Loading the test dataset...")
    test_data = load_data(test_dir, args.normalised, args.denoised)
    data_loader = {"test": test_data}

    #Looping over the K folds
    for k in range(args.kfold):

        #Initialising the model
        trained_model = glob.glob(os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_fold{}*.pkl".format(k)))[0]
        if args.verbose:
            print("Loading model: {}...\n".format(os.path.basename(trained_model)))
        model = ResNet34().to(device)
        model.load_state_dict(torch.load(trained_model), strict=False)

        #Getting the class weights
        if args.weight:
            weights = get_weights(test_dir, args.normalised, args.denoised, args.verbose)
        else:
            weights = [1., 1.]

        #Settings for training the model
        weights = torch.FloatTensor(weights).to(device)
        loss = nn.CrossEntropyLoss(weight=weights)

        #Making the predictions
        make_preds(model, loss, data_loader, k, fout, device, args.verbose)

    #Scoring the evaluations
    score(fout)



if __name__ == '__main__':
    main()
