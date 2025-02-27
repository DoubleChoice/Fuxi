import torch.nn as nn
import torch

class FCLayer(nn.Module):
    def __init__(self,input_channels,input_H,input_W,out_channels,output_H,output_W):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Linear(input_channels, out_channels)
        self.fc2 = nn.Linear(input_W, output_W)
        self.fc3 = nn.Linear(input_H, output_H)
        self.dropout = nn.Dropout(p=0.9)
    def forward(self, x):
        # [batch_size,1536,180,360]
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1)  # [batch_size,180,360,1536]
        x = self.fc1(x)     # [batch_size,180,360,5]
        x = self.dropout(x)
        x = x.permute(0,3,1,2)  # [batch_size,5,180,360]
        x = self.fc2(x)     # [batch_size,5,180,1440]
        x = self.dropout(x)
        x = x.permute(0,1,3,2)  # [batch_size,5,1440,180]
        x = self.fc3(x)    # [batch_size,5,1440,721]
        x = self.dropout(x)
        x = x.permute(0,1,3,2)  # [batch_size,5,721,1440]
        return x