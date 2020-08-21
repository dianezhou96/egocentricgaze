import torch
import torch.nn as nn
import torch.optim as optim
from video_to_data import *
import matplotlib.pyplot as plt



class SaliencyShiftedGridsNet(nn.Module):

    def __init__(self, N):
        super(SaliencyShiftedGridsNet, self).__init__()

        # First 5 layers of AlexNet
        self.alexnet = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Additional convolutional layer to turn into saliency map
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 5 shifted grids
        self.shifted_grid_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.Softmax(dim=1)
        )
        self.shifted_grid_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.Softmax(dim=1)
        )
        self.shifted_grid_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.Softmax(dim=1)
        )
        self.shifted_grid_4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.Softmax(dim=1)
        )
        self.shifted_grid_5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.alexnet(x)
        x = self.classifier(x)
        outs = [
            self.shifted_grid_1(x),
            self.shifted_grid_2(x),
            self.shifted_grid_3(x),
            self.shifted_grid_4(x),
            self.shifted_grid_5(x),
        ]
        return outs

