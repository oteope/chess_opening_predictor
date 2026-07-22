import torch
import torch.nn as nn


class ChessMLP(nn.Module):

    def __init__(self,
                 input_size=774,
                 output_size=3):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

             nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

             nn.Linear(128, 64),
             nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, output_size)

            )


    def forward(self, x):

        return self.network(x)