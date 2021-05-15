import torch
torch.manual_seed(123)
import torch.nn as nn

class RCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super(RCNN, self).__init__()

        self._extractor = nn.Sequential(
            # Bx1x256x256
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # Bx64x256x256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Bx64x128x128
            
            # Bx64x128x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # Bx128x128x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Bx128x64x64

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # Bx256x64x64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4), # Bx256x16x16

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # Bx512x16x16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8) # Bx512x2x2
        )

        self._rnnModule = nn.Sequential(
                 nn.GRU(512, 512, batch_first=False,bidirectional=True),
                #nn.LSTM(512, 512, batch_first=False, bidirectional=True),
                )

        self._classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(in_features=512, out_features=128),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(in_features=128, out_features=8))
        self.apply(self._init_weights)

    def forward(self, x):
        x = self._extractor(x) # Bx512x2x2
#         x = x.permute(3,0,1,2)
        x = x.view(x.size(0), -1, x.size(1))  # Bx4x512
        x, hn = self._rnnModule(x) # Bx4x1024
#         x = x.permute(1, 2, 0)
        x = x.view(x.size(0), -1)
        score = self._classifier(x)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)