import torch
import torch.nn as nn


class ChessMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

        nn.Linear(774, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(256, 3)

    )

    def forward(self, x):

        return self.network(x)