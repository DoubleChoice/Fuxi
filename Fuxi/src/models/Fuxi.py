import torch
import torch.nn as nn
from torch import optim
from src.models.modules.L1_loss import L1Loss
from src.models.modules.DownBlock import DownBlock
from src.models.modules.MLP import FCLayer
from src.models.modules.SwinTransformerV2 import SwinTransformerBlock
from src.models.modules.UpBlock import UpBlock
import pytorch_lightning as pl

class Fuxi(pl.LightningModule):
    def __init__(self, input_channels=5, output_channels=5, batch_size=3):
        super(Fuxi, self).__init__()
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=1536, kernel_size=(2, 4, 4),
                               stride=(2, 4, 4))

        self.cube_embedding_layer_norm = nn.LayerNorm([1, 180, 360])

        self.down_block = DownBlock(input_channels=1536, out_channels=1536)

        self.swin_transformer_blocks = nn.Sequential(
            *[SwinTransformerBlock(dim=1536, input_resolution=(90, 180), num_heads=2, window_size=2) for _ in range(1)])

        self.up_block = UpBlock(input_channels=1536, out_channels=1536)

        self.fc_layer = FCLayer(input_channels=1536, input_H=180, input_W=360, out_channels=output_channels, output_H=721,
                                output_W=1440)

    def forward(self, x):
        # [batch_size, p, 5, 721, 1440]
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, 5, p, 721, 1440]
        x = self.relu(self.conv1(x))  # [batch_size, 1536, 1, 180, 360]

        x = self.cube_embedding_layer_norm(x)  # [batch_size, 1536, 1, 180, 360]

        x = x.squeeze(2)  # [batch_size, 1536, 180, 360]

        x = self.down_block(x)  # [batch_size, 1536, 90, 180]

        x = x.permute(0, 2, 3, 1)  # [batch_size, 90, 180,1536]
        add_x = x  # add operation x [batch_size, 90,180,1536]
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)  # [batch_size, 90*180,1536]
        x = self.swin_transformer_blocks(x)
        x = x.view(B, H, W, C)
        x += add_x  # [batch_size, 90,180,5]
        x = x.permute(0, 3, 1, 2)  # [batch_size,1536,90,180]
        x = self.up_block(x)  # [batch_size,1536,180,360]
        x = self.fc_layer(x)  # [batch_size,5,721,1440]
        return x

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.00025, betas=(0.9, 0.95), weight_decay=0.1,eps=1e-5)

    def training_step(self, batch, batch_id):
        x, y = batch
        y_hat = self(x)
        loss = L1Loss()
        return loss(y_hat, y)