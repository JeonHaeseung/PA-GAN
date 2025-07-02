import torch
import torch.nn as  nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        # self.fc1 = nn.Linear(64*4*4, 200) # mnist
        # self.fc1 = nn.Linear(64*5*5, 200)   # cifar10
        self.fc1 = nn.Linear(64*21*21, 200)   # stl10
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x
    

def ConvNet4(num_classes, channels):
    return ConvNet(num_classes, channels)