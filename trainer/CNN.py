import torch
torch.manual_seed(123)
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, dropout=0.2):
        super(CNN, self).__init__()

        self._extractor = nn.Sequential(
            # Bx1x128x128
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # Bx64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Bx64x64x64
            
            # Bx64x128x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # Bx128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Bx128x32x32

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # Bx256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4), # Bx256x8x8

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # Bx512x8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8) # Bx512x1x1
        )

        self._rnnModule = nn.Sequential(
                 nn.GRU(512, 512, batch_first=True,bidirectional=False),
                #nn.LSTM(512, 512, batch_first=False, bidirectional=True),
                )
        
        self._classifier = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(in_features=256, out_features=128),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(in_features=128, out_features=8))
        self.apply(self._init_weights)

    def forward(self, x):
        x = self._extractor(x) # Bx512x1x1
        x = x.view(x.size(0), -1, x.size(1))  # Bx1x512
        x, hd = self._rnnModule(x)
        x = x.view(x.size(0), -1)
        score = self._classifier(x) # Bx8
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)