import torch.nn as nn
import torch.optim as optim
import torch
from src.models.Fuxi import Fuxi

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = Fuxi(input_channels=5).to(device).bfloat16()
# [batch_size,p,721,1440,5]
data = torch.randn((1, 2,5, 721, 1440)).to(device).bfloat16()
output = model(data)
print(output.shape)

