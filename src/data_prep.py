#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import wfdb
from sklearn.model_selection import train_test_split


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('-o', '--output', type=str, help='Name of the output data directory')
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')
    parser.set_defaults(download=True)

    #Options for preprocessing
    parser.add_argument('-t', '--test', type=float, help='Ratio of samples that is included in the test dataset')

    #Printing arguments to the command line
    args = parser.parse_args()
    print('Called with args:')
    print(args)

    return args


def split_data(data):

    #Initialising variables
    Xs, ys = [], []
    valid_symbols = ["N", "L", "R", "e", "j", "V", "E"]

    #Looping over the patients of the dataset
    for s in data:
        X, y = [], []
        signals = wfdb.rdsamp(s)[0]
        symbols = wfdb.rdann(s, 'atr').symbol
        samples = wfdb.rdann(s, 'atr').sample

        #Slicing the signal data
        for i in range(len(symbols)):
            start = samples[i] - 360
            end = samples[i] + 360

            #Classifying heartbeats as Regular or Ventricular Ectropic
            if symbols[i] in valid_symbols and start >= 0 and end <= len(signals) and len(signals[start:end]) == 720:
                X.append(signals[start:end])
                if symbols[i] == "V" or symbols[i] == "E":
                    y.append(1)
                else:
                    y.append(0)

        #Appending the extracted signal to the explanatory and response variable lists
        Xs.append(np.array(X))
        ys.append(np.array(y))

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(Xs), np.concatenate(ys), train_size=0.2, random_state=0, stratify=np.concatenate(ys))

    return X_train, X_test, y_train, y_test


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))

    #Downloading the MIT-BIH Arrhythmia Database dataset
    if args.download:
        download_dir = os.path.join(path, "data", "raw")
        wfdb.dl_database('mitdb', dl_dir=download_dir)

    #Removing subjects for which a modified limb lead II electrode was not used
    MLII_subjects = [subject[:-4] for subject in glob.glob(f"{path}/data/raw/*.dat") if wfdb.rdsamp(subject[:-4])[1]["sig_name"][0] == "MLII"]

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = split_data(MLII_subjects)

    #Saving the train and test datasets
    np.save(os.path.join(path, "data", "train", "X_train.npy"), X_train)
    np.save(os.path.join(path, "data", "train", "y_train.npy"), y_train)

    np.save(os.path.join(path, "data", "test", "X_test.npy"), X_test)
    np.save(os.path.join(path, "data", "test", "y_test.npy"), y_test)


if __name__ == '__main__':
    main()
