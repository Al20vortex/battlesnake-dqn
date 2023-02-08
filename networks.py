from collections import OrderedDict
import torch
import torch.nn as nn

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps"
# device = torch.device('cpu')
print(f"Using {device} device")

class QNetworkModel(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 4, 2)
        self.relu2 = nn.ReLU()
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(16, num_outputs)
        # self.relu2 = nn.ReLU()
        # self.linear2 = nn.Linear(6, num_outputs)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu1(self.flatten1(x))
        x = self.linear1(x)
        # x = self.relu2(self.linear1(x))
        # x = self.linear2(x)
        return x
