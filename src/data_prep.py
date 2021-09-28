#!/usr/bin/env python

import argparse
import glob
import numpy as np
import pywt
import os
import wfdb
from collections import Counter
from skimage.restoration import denoise_wavelet
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
    print("Remaining number of patients: {}".format(len(MLII_subjects)))

    #Loading the data
    X, y = load_data(MLII_subjects, args.test)

    #Normalizing the data
    if args.norm:
        if args.verbose:
            print("\nNormalising the dataset...")
        X = (X - np.mean(X) / np.std(X))
        suffix = "_normalised"

    #Splitting the data into train and test
    if args.verbose:
        print("Splitting the dataset and into train ({}%) and test ({}%)...".format(str(int((1-float(args.test))*100)), str(int(float(args.test)*100))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, stratify=y)

    #Denoising the training data
    if args.denoise:
        if args.verbose:
            print("Denoising the training dataset...")
        level = int(np.floor(np.log(len(X_train))/2.0))
        X_denoise = denoise_wavelet(X_train[:,0], wavelet=args.wavelet, mode='soft', wavelet_levels=level, method='BayesShrink', rescale_sigma='True')
        suffix = "_denoised"

    #Saving the train and test datasets
    if args.verbose:
        print("\nSaving the training data in: {}".format(os.path.join(path, "data", "train")))
        print("Saving the test data in: {}".format(os.path.join(path, "data", "test")))

    np.save(os.path.join(path, "data", "train", "X_train{}.npy".format(suffix)), X_train)
    np.save(os.path.join(path, "data", "train", "y_train.npy"), y_train)

    np.save(os.path.join(path, "data", "test", "X_test{}.npy".format(suffix)), X_test)
    np.save(os.path.join(path, "data", "test", "y_test.npy"), y_test)


if __name__ == '__main__':
    main()
