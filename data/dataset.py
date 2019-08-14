import joblib
import torch
import numpy as np
from torch.utils.data import Dataset

DATA_DIR = '/insert/data/directory/here'


class MIMIC(Dataset):
    """
    Dataset for MIMIC
    """
    def __init__(self, mode, image=False):
        """
            mode: 'train, 'val', 'test'
            data: a tuple consisting of a numpy array (X) and a list (y).
                len(y)=X.shape[0]=size of data set/ number of patients/admissions
                X.shape[1:]=(48,76)=(T,d)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if mode == 'train':
            self.x, self.y = joblib.load('{}/mimic_train.pkl'.format(DATA_DIR))
        elif mode == 'val':
            self.x, self.y = joblib.load('{}/mimic_validation.pkl'.format(DATA_DIR))
        elif mode == 'test':
            self.x, self.y = joblib.load('{}/mimic_test.pkl'.format(DATA_DIR))
        elif mode == 'trainval':
            xtr, ytr = joblib.load('{}/mimic_train.pkl'.format(DATA_DIR))
            xval, yval = joblib.load('{}/mimic_validation.pkl'.format(DATA_DIR))
            self.x = np.concatenate((xtr, xval))
            self.y = np.concatenate((ytr, yval))
        else:
            raise ValueError('Mode {} not proper'.format(mode))
        self.x = np.swapaxes(self.x, 1, 2)
        if image:
            self.x = np.expand_dims(self.x, 1)
        self.x = torch.from_numpy(self.x).to(dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ImbalancedMNIST(Dataset):
    def __init__(self, mode):
        if mode == 'trainval':
            self.x, self.y = joblib.load('{}/mnist_imbalance_trainval.pkl'.format(DATA_DIR))
        elif mode == 'test':
            self.x, self.y = joblib.load('{}/mnist_imbalance_test.pkl'.format(DATA_DIR))
        else:
            raise ValueError('mode == {} not proper value'.format(mode))
        self.x = torch.from_numpy(self.x).to(dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
