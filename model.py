import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Parameter


class Mars_Spectrometry_Model(nn.Module):

    def __init__(self, num_features, num_classes):
        super(Mars_Spectrometry_Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.apply(self.init_weights_xavier)

    def init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.layers(x)
