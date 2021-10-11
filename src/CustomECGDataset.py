#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, X_path, y_path, sampling=False):
        if sampling:
            signal, labels = np.load(X_path), np.load(y_path)
            df = pd.DataFrame(signal)
            df['y'] = labels
            samples = random.sample(list(df[df['y'] == 0].index), df[df['y'] == 1].shape[0]) + list(df[df['y'] == 1].index)
            sampled_df = df[df.index.isin(samples)]

            self.labels = sampled_df['y'].to_numpy()
            self.signal = sampled_df.loc[:, df.columns != 'y'].to_numpy()
        else:
            self.signal = np.load(X_path)
            self.labels = np.load(y_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signal[None, idx].astype(np.float32)
        label = self.labels[idx]
        return signal, label
