import torch
import torch.nn.functional as F
from torch import nn, Tensor

class femnistMLP(nn.Module):

    def __init__(self, num_classes=10,w_h=10):
        super(femnistMLP, self).__init__()
        self.linear = nn.Linear(28 * 28, w_h)
        self.fc = nn.Linear(w_h, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc = nn.Linear(84, num_classes)

    def forward(self, img):
        feature = self.conv(img)
        output = self.linear(feature.view(img.shape[0], -1))
        output = self.fc(output)
        return output
