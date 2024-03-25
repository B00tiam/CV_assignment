# LeNet5 models: 1 baseline, 4 variants
# Definition of each model

import torch
import torch.nn as nn

class LeNet5_baseline(nn.Module):
    def __init__(self):
        super(LeNet5_baseline, self).__init__()
        # The first convolutional layer, activation function and pooling layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # maxpool layer
        # The second convolutional layer, activation function and pooling layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Three fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # size of fully connected layer 1
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)  # softmax layer

        # initialization of bias
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    # forward func
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
