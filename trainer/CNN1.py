import torch
torch.manual_seed(123)
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, dropout=0.2):
        super(CNN1, self).__init__()

        self._extractor = nn.Sequential(
            # Bx1x16384
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # Bx64x16384
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4), # Bx64x4096
            
            # Bx64x128x128
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # Bx128x4096
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4), # Bx128x1024

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # Bx256x1024
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8), # Bx256x128
        )

        # Bx256x1024
        self._rnnModule = nn.Sequential(
            nn.GRU(128, 128, batch_first=True, bidirectional=False),
        )
        
        self._classifier = nn.Sequential(nn.Linear(in_features=32768, out_features=2048),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(in_features=2048, out_features=128),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(in_features=128, out_features=8))
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1) # Bx1x16384
        x = self._extractor(x) # Bx256x128
#         x = x.view(x.size(0), -1, x.size(1))
#         x = x.permute(0, 2, 1)
        x, hd = self._rnnModule(x) # Bx128x256
        x = x.reshape(x.size(0), -1)
        x = self._classifier(x) # Bx8
                
        return x

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)