#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.signal = np.load(X_path)
        self.labels = np.load(y_path)

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        signal = self.signal[None, idx].astype(np.float32)
        label = self.labels[idx]
        return signal, label
