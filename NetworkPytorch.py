import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Module):

    def __init__(self, sizes):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], sizes[2]),
            nn.Softmax()
        )

        def forward(self, x):
            return self.seq(x)



