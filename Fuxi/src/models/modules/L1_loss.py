import torch.nn as nn
import torch


class L1Loss(nn.Module):
    def __init__(self, ):
        super(L1Loss, self).__init__()

    def forward(self, out, target):
        B, C, H, W = out.shape
        dif = torch.abs(out - target)
        latitude_indices = torch.arange(H, dtype=torch.float16)
        a = torch.exp(-torch.abs(latitude_indices - H // 2) / (H // 2))
        if a.device != dif.device:
            a = a.to(dif.device)
        dif = dif * a.view(-1, 1)
        loss = dif.sum() / (B*C*H*W)
        return loss
