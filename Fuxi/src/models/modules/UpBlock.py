import torch.nn as nn
import torch
from src.models.modules.ResidualBlock import ResidualBlock

class UpBlock(nn.Module):
    def __init__(self, input_channels=10, out_channels = 10):
        super(UpBlock, self).__init__()
        self.relu = nn.ReLU()
        self.residual_block = ResidualBlock(input_channels=input_channels,out_channels=out_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=out_channels, kernel_size=2, stride=2)



    def forward(self, x):
        x = self.residual_block(x)  #[batch_size, 10,90,180]
        x = self.conv1(x)   #[batch_size, 10,180,360]
        return x
