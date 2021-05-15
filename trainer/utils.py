import torch.nn
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np

import deepdish as dd
import os 
import subprocess

STORAGE_BUCKET = 'fma_preprocessed_data'
X_DATA_PATH = 'X_3.h5'
Y_DATA_PATH = 'Y_3.h5'
LOCAL_PATH = '/tmp'

class FMADataset(Dataset):
    def __init__(self, X, Y, mode='training'):
        if mode not in ('training', 'validation', 'test'):
            raise Error('mode must be training, validation, or test.')
        self.mode = mode
        self.x = X[mode]
        self.y = Y[mode]
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # ndarray contains negative strides
        # to fix, we flip a copy, and flip again
        image = torch.from_numpy(np.flip(self.x[index], axis=0).astype('float64').copy())
        image = torch.flip(image, dims=(0,))
#         image = transforms.Normalize(256, 256)(image)
        label = self.y[index]
        return image.float(), label
    
def load_FMADatasets(X_mode, Y_mode):
    """Download FMA Datasets and return
    
    Returns X, Y 
    """
    local_x = os.path.join(LOCAL_PATH, X_DATA_PATH)
    local_y = os.path.join(LOCAL_PATH, Y_DATA_PATH)
    subprocess.call([
        'gsutil', 'cp',
        # Storage path
        os.path.join('gs://', STORAGE_BUCKET, X_DATA_PATH),
        # Local path
        local_x
    ])
    subprocess.call([
        'gsutil', 'cp',
        # Storage path
        os.path.join('gs://', STORAGE_BUCKET, Y_DATA_PATH),
        # Local path
        local_y
    ])
    X = dd.io.load(local_x)
    Y = dd.io.load(local_y)
    
    mean = np.mean(X['training'])
    std = np.std(X['training'])
    
    for k in X.keys():
        X[k] = (X[k] - mean) / std
    
    X_data = FMADataset(X, Y, mode=X_mode)
    Y_data = FMADataset(X, Y, mode=Y_mode)
    
    return X_data, Y_data

if __name__=='__main__':
    print(os.listdir())