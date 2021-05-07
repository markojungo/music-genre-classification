import torch.utils.data as data

class FMADataset(data.Dataset):
    """Custom Dataset for FMA data
    
    """
    def __init__(self, mode='train'):