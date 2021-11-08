#!/usr/bin/env python

import argparse
import glob
import numpy as np
import pywt
import os
import wfdb

from collections import Counter
from skimage.restoration import denoise_wavelet
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('-o', '--output', type=str, help='Name of the output data directory')
    parser.add_argument('-d', '--download', type=str2bool, nargs='?', const=True, default=False, help='Option to download the raw data')
    parser.add_argument('-v', '--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')

    #Options for preprocessing
    parser.add_argument('--test', type=float, nargs='?', default=0.2, help='Ratio of samples that is included in the test dataset')
    parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False, help='Option to normalize the dataset')
    parser.add_argument('--denoise', type=str2bool, nargs='?', const=True, default=False, help='Option to denoise the dataset')
    parser.add_argument('--wavelet', type=str, default='sym8', help='Type of wavelet to perform denoising (default: sym8)')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')

    #Options for class imbalance
    parser.add_argument('--weight', type=str2bool, nargs='?', const=True, default=False, help='Rescale the weights of the loss function to alleviate class imbalance')
    parser.add_argument('--sampling', type=str2bool, nargs='?', const=True, default=False, help='Randomly sample heartbeats from the majority class to match the number of the minority class')

    #Printing arguments to the command line
    args = parser.parse_args()

    #Checking arguments
    assert args.wavelet in pywt.wavelist(), 'Specified wavelet is not available'

    print('Called with args:')
    print(args)

    return args


def get_stats(path):

    #Initialising variables
    subjects = [subject for subject in glob.glob(f"{path}/data/raw/*.dat")]
    data = [subject[:-4] for subject in subjects]
    signal_names, symbol_types = [], []
    normal, supraventricular, ventricular, fusion, unknown = 0, 0, 0, 0, 0
    AAMI_class = {"N": ["N", "L", "R", "e", "j"],
                  "S": ["A", "a", "J", "S"],
                  "V": ["V", "E"],
                  "F": ["F"],
                  "Q": ["/", "f", "Q"]}

    #Displaying the number of subjects
    print("\nNumber of subjects: {}".format(len(subjects)))

    #Looping over the patients of the dataset
    for patient in data:
        info = wfdb.rdsamp(patient)[1]
        symbols = wfdb.rdann(patient, 'atr').symbol

        #Saving signal source and signal type data
        signal_names.append(info["sig_name"])
        symbol_types.append(symbols)

    #Displaying the used electrodes
    print("\nElectrode combinations used: ")
    electrode_count = Counter([" ".join(sn) for sn in signal_names]).most_common()
    for ec in electrode_count:
        print(ec[0], ec[1])

    #Displaying the type of heartbeats
    print("\nRegistered heartbeats: ")
    heartbeats = [x for st in symbol_types for x in st]
    for hbeat in heartbeats:
        if hbeat in AAMI_class["N"]:
            normal += 1
        elif hbeat in AAMI_class["S"]:
            supraventricular += 1
        elif hbeat in AAMI_class["V"]:
            ventricular += 1
        elif hbeat in AAMI_class["F"]:
            fusion += 1
        elif hbeat in AAMI_class["Q"]:
            unknown +=1
    print("N {}".format(str(normal)))
    print("S {}".format(str(supraventricular)))
    print("V {}".format(str(ventricular)))
    print("F {}".format(str(fusion)))
    print("Q {}".format(str(unknown)))


def load_data(data, testsize):

    #Initialising variables
    Xs, ys = [], []

    #Looping over the patients of the dataset
    for s in data:
        X, y = [], []
        info = wfdb.rdsamp(s)[1]
        signals = wfdb.rdsamp(s)[0][:, 0]
        symbols = wfdb.rdann(s, 'atr').symbol
        samples = wfdb.rdann(s, 'atr').sample

        #Only keeping patients for which MLII and V1 electrodes were used
        if "MLII" in info["sig_name"] and "V1" in info["sig_name"]:

            #Slicing the signal data
            for i in range(len(symbols)):
                start = samples[i] - int(info["fs"])
                end = samples[i] + int(info["fs"])

                #Saving data from the 2 specified heartbeat classes
                if symbols[i] in "NV" and start >= 0 and end <= len(signals) and len(signals[start:end]) == int(info["fs"])*2:
                    X.append(signals[start:end])
                    if symbols[i] == "N":
                        y.append(0)
                    elif symbols[i] == "V":
                        y.append(1)

            #Appending the extracted signal to the explanatory and response variable lists
            if len(X) > 0:
                Xs.append(np.array(X))
                ys.append(np.array(y))

    return (np.vstack(Xs), np.concatenate(ys))


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    suffix = ""
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))

    #Downloading the MIT-BIH Arrhythmia Database dataset
    if args.download:
        if args.verbose:
            print("\nDownloading the dataset...")
        download_dir = os.path.join(path, "data", "raw")
        wfdb.dl_database('mitdb', dl_dir=download_dir)

    #Getting an overview of the data
    if args.verbose:
        get_stats(path)

    #Removing subjects for which modified limb lead II (MLII) and modified limb V1 (V1) electrodes were not used
    if args.verbose:
        print("\nRemoving subjects for which MLII and V1 electrodes were not used...")
    MLII_subjects = [subject[:-4] for subject in glob.glob(f"{path}/data/raw/*.dat") if wfdb.rdsamp(subject[:-4])[1]["sig_name"][0] == "MLII" and wfdb.rdsamp(subject[:-4])[1]["sig_name"][1] == "V1"]
    if args.verbose:
        print("Remaining number of patients: {}".format(len(MLII_subjects)))

    #Loading the data
    X, y = load_data(MLII_subjects, args.test)

    #Splitting the data into (temporary) train and test
    if args.verbose:
        print("Splitting the dataset and into train ({}%) and test ({}%)...".format(str(int((1-float(args.test))*100)), str(int(float(args.test)*100))))
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=args.test, stratify=y)

    #Normalizing the test dataset
    if args.norm:
        X_test = (X_test - np.mean(X_test) / np.std(X_test))
        suffix = "_normalised"

    #Denoising the test dataset
    if args.denoise:
        level_test = int(np.floor(np.log(len(X_test))/2.0))
        X_test = denoise_wavelet(X_test[:, 0:], wavelet=args.wavelet, mode='soft', wavelet_levels=level_test, method='BayesShrink', rescale_sigma='True')
        if suffix:
            suffix += "_denoised"
        else:
            suffix = "_denoised"

    #Splitting the temporary training dataset into K training and validation datasets
    if args.verbose:
        print("\nSplitting the temporary training dataset into {} folds...".format(args.kfold))

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    for idx, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
        if args.verbose:
            print("TRAIN:", train_index, "VAL:", val_index, idx)

        X_train, X_val = X_temp[train_index], X_temp[val_index]
        y_train, y_val = y_temp[train_index], y_temp[val_index]

        #Taking random samples from the majority class equal to the number in the minority class
        if args.sampling:
            X_major = X_train[np.where(y_train==0)]
            y_major = y_train[np.where(y_train==0)]
            X_minor = X_train[np.where(y_train==1)]
            y_minor = y_train[np.where(y_train==1)]

            if args.verbose:
                print("\nTaking random samples from the majority class ({})...".format(X_major.shape[0]))

            X_major_sampled = X_major[np.random.choice(X_major.shape[0], X_minor.shape[0], replace=False)]
            y_major_sampled = y_major[np.random.choice(y_major.shape[0], y_minor.shape[0], replace=False)]
            X_train = np.concatenate((X_major_sampled, X_minor), axis=0)
            y_train = np.concatenate((y_major_sampled, y_minor), axis=0)

            if args.verbose:
                print("Sampled majority class samples: {}".format(X_major_sampled.shape[0]))

        #Normalizing the train and validation datasets
        if args.norm:
            if args.verbose:
                print("\nNormalising the datasets...")
            X_train = (X_train - np.mean(X_train) / np.std(X_train))
            X_val = (X_val - np.mean(X_val) / np.std(X_val))

        #Denoising the train and validation datasets
        if args.denoise:
            if args.verbose:
                print("Denoising the datasets...")
            level_train = int(np.floor(np.log(len(X_train))/2.0))
            level_val = int(np.floor(np.log(len(X_val))/2.0))
            X_train = denoise_wavelet(X_train[:, 0:], wavelet=args.wavelet, mode='soft', wavelet_levels=level_train, method='BayesShrink', rescale_sigma='True')
            X_val = denoise_wavelet(X_val[:, 0:], wavelet=args.wavelet, mode='soft', wavelet_levels=level_val, method='BayesShrink', rescale_sigma='True')

        #Saving the train and validation datasets
        np.save(os.path.join(path, "data", "train", "X_train{}_{}.npy".format(suffix, idx)), X_train)
        np.save(os.path.join(path, "data", "train", "y_train_{}.npy".format(idx)), y_train)

        np.save(os.path.join(path, "data", "val", "X_val{}_{}.npy".format(suffix, idx)), X_val)
        np.save(os.path.join(path, "data", "val", "y_val_{}.npy".format(idx)), y_val)

    #Saving the test dataset
    np.save(os.path.join(path, "data", "test", "X_test{}.npy".format(suffix)), X_test)
    np.save(os.path.join(path, "data", "test", "y_test.npy"), y_test)

    if args.verbose:
        print("\nSaved the training datasets to: {}".format(os.path.join(path, "data", "train")))
        print("Saved the validation datasets to: {}".format(os.path.join(path, "data", "val")))
        print("Saved the test dataset to: {}".format(os.path.join(path, "data", "test")))


if __name__ == '__main__':
    main()
