import torch.nn as nn
import torch
from src.models.modules.ResidualBlock import ResidualBlock

class DownBlock(nn.Module):
    def __init__(self, input_channels=5, out_channels = 5):
        super(DownBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3),stride=2,padding=1)

        self.residual_block = ResidualBlock(input_channels=input_channels,out_channels=out_channels)


    def forward(self, x):
        # [batch_size, 5, 180, 360]
        x = self.relu(self.conv1(x))    #[batch_size, 5,90,180]

        x = self.residual_block(x)  #[batch_size, 5,90,180]
        return x
