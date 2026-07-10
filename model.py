"""
CNN speed-estimation model.

Input : a 128x128x3 optical-flow image (HSV motion encoded as BGR).
Output: a single number = estimated ego-vehicle speed.

Architecture vendored from Yash Shah's comma.ai speed-challenge solution
(https://github.com/shahyash10/speedchallenge) so the bundled pretrained
weights (weights/Model.pt) load without any external download.
"""

import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 8)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 24, 7)
        self.conv3_bn = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 36, 4)
        self.conv4_bn = nn.BatchNorm2d(36)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # convolution layers
        x = self.relu(self.pool(self.conv1_bn(self.conv1(x))))
        x = self.relu(self.pool(self.conv2_bn(self.conv2(x))))
        x = self.relu(self.pool(self.conv3_bn(self.conv3(x))))
        x = self.relu(self.pool(self.conv4_bn(self.conv4(x))))
        # fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x
