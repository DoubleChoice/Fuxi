import torch
from torch.utils.data import DataLoader
from data.DataSet import WeatherDataset
from src.models.modules.L1_loss import L1Loss
from src.models.Fuxi import Fuxi
import torch.optim as optim
import wandb
import pytorch_lightning as pl

dataset = WeatherDataset('/data/xlw/X_p_{}.pt'.format(1),'/data/xlw/Y_p_{}.pt'.format(1))
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = Fuxi()
trainer = pl.Trainer(max_epochs=100,precision="bf16",devices=2)
trainer.fit(model, train_dataloader)