from collections import OrderedDict
import torch
import torch.nn as nn

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device('cpu')
print(f"Using {device} device")


class QNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 24, 2)
        self.relu2 = nn.ReLU()
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(96, 8)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.flatten1(x)
        x = self.relu2(x)
        x = self.linear1(x)
        self.relu3(x)
        x = self.linear2(x)
        return x
