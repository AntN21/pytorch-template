from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import wfdb


class PTBDataset(Dataset):
    def __init__(self,X,y,axis=None) -> None:
        super().__init__()
        if axis is None:
            self.X = X
        else:
            self.X = X[:,axis,:]

        self.y = y

        self.X = self.X.float()
        self.y = self.y.float()
        self.compute_normalizers()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return the features and corresponding label at the given index
        return self.X[idx], self.y[idx]

    def compute_normalizers(self):
        """
        Computes the mean and standard deviation of the ECG and HRV time series.
        """
        self.X_mean = torch.mean(self.X)
        self.X_std = torch.std(self.X)

    def get_num_classes(self):
        return 5
    
class PTBDataLoader(BaseDataLoader):
    def __init__(self, data_dir,
                lead,
                batch_size,
                shuffle=True,
                validation_split=0.0,
                num_workers=1,
                training=True,
                sampling_rate=100,
                seed = None):
        
        X = torch.load(os.path.join(data_dir,f'X{sampling_rate}.pth'))
        y = torch.load(os.path.join(data_dir,f'y.pth'))

        self.dataset = PTBDataset(X,y,axis=lead)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
