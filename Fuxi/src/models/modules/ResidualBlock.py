import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.gn = nn.GroupNorm(64, out_channels)

    def forward(self, x):
        residual_x = x  #[batch_size,C,H,W]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.silu(self.gn(x))
        x = x + residual_x
        return x
