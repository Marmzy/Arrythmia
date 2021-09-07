#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import wfdb
from collections import Counter
from sklearn.model_selection import train_test_split


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('-o', '--output', type=str, help='Name of the output data directory')
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')
    parser.set_defaults(download=True)

    #Options for preprocessing
    parser.add_argument('-b', '--hbeat', type=str, help='AAMI heartbeat symbols to classify')
    parser.add_argument('-t', '--test', type=float, nargs='?', default=0.2, help='Ratio of samples that is included in the test dataset')

    #Printing arguments to the command line
    args = parser.parse_args()
    print('Called with args:')
    print(args)

    return args


def get_stats(data):

    #Initialising variables
    signal_names, symbol_types = [], []
    normal, supraventricular, ventricular, fusion, unknown = 0, 0, 0, 0, 0
    AAMI_class = {"N": ["N", "L", "R", "e", "j"],
                  "S": ["A", "a", "J", "S"],
                  "V": ["V", "E"],
                  "F": ["F"],
                  "Q": ["/", "f", "Q"]}

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


def split_data(data, hbeat_class, testsize):

    #Initialising variables
    Xs, ys = [], []

    #Looping over the patients of the dataset
    for s in data:
        X, y = [], []
        info = wfdb.rdsamp(s)[1]
        signals = wfdb.rdsamp(s)[0]
        symbols = wfdb.rdann(s, 'atr').symbol
        samples = wfdb.rdann(s, 'atr').sample

        #Only keeping patients for which MLII and V1 electrodes were used
        if "MLII" in info["sig_name"] and "V1" in info["sig_name"]:

            #Slicing the signal data
            for i in range(len(symbols)):
                start = samples[i] - int(info["fs"])
                end = samples[i] + int(info["fs"])

                #Saving data from the 2 specified heartbeat classes
                if symbols[i] in hbeat_class and start >= 0 and end <= len(signals) and len(signals[start:end]) == int(info["fs"])*2:
                    X.append(signals[start:end])
                    if symbols[i] == hbeat_class[0]:
                        y.append(0)
                    elif symbols[i] == hbeat_class[1]:
                        y.append(1)

            #Appending the extracted signal to the explanatory and response variable lists
            if len(X) > 0:
                Xs.append(np.array(X))
                ys.append(np.array(y))

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(Xs), np.concatenate(ys), test_size=testsize, random_state=0, stratify=np.concatenate(ys))

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

    #Getting an overview of the data
    get_stats(MLII_subjects)

    #Splitting the data into train and test
    X_train, X_test, y_train, y_test = split_data(MLII_subjects, args.hbeat, args.test)

    #Saving the train and test datasets
    np.save(os.path.join(path, "data", "train", "X_train.npy"), X_train)
    np.save(os.path.join(path, "data", "train", "y_train.npy"), y_train)

    np.save(os.path.join(path, "data", "test", "X_test.npy"), X_test)
    np.save(os.path.join(path, "data", "test", "y_test.npy"), y_test)


if __name__ == '__main__':
    main()
