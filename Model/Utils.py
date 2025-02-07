import os
import torch
import numpy as np
import torch.nn as nn
import scipy.io as io
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset


class prepareData(Dataset):
    def __init__(self, dir, fs):
        folders = os.listdir(dir)
        self.filePaths = []
        self.fs = fs
        for folder in folders:
            files = os.listdir(os.path.join(dir, folder))
            self.filePaths.extend(dir+folder+'/'+file for file in files)
            pass
        pass

    def __len__(self):
        return len(self.filePaths)
        pass

    def __getitem__(self, item):
        itemPath = self.filePaths[item]

        if itemPath.__contains__('Classone'):
            label = [1,0]
        elif itemPath.__contains__('Classtwo'):
            label = [0,1]
            pass


        data = io.loadmat(itemPath)['Data'].T

        delta = torch.FloatTensor(self.scale_input(self.butter_bandpass_filter(data, 0.01, 4, self.fs, order=5).copy())).unsqueeze(2)
        theta = torch.FloatTensor(self.scale_input(self.butter_bandpass_filter(data, 4, 8, self.fs, order=5).copy())).unsqueeze(2)
        alpha = torch.FloatTensor(self.scale_input(self.butter_bandpass_filter(data, 8, 13, self.fs, order=5).copy())).unsqueeze(2)
        beta = torch.FloatTensor(self.scale_input(self.butter_bandpass_filter(data, 13, 25, self.fs, order=5).copy())).unsqueeze(2)
        gamma = torch.FloatTensor(self.scale_input(self.butter_bandpass_filter(data, 25, 49, self.fs, order=5).copy())).unsqueeze(2)

        data = torch.cat([delta, theta, alpha, beta, gamma])
        label = torch.FloatTensor(label)

        return data, label
        pass

    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def scale_input(self, input):
        mean = np.mean(input)
        std = np.std(input)

        output = (input - mean)/std
        return output
        pass
    pass