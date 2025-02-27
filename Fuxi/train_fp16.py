import torch
from torch.utils.data import DataLoader
from data.DataSet import WeatherDataset
from src.models.modules.L1_loss import L1Loss
from src.models.Fuxi import Fuxi
import torch.optim as optim


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
dataset = WeatherDataset('/data/xlw/data_stream-oper_stepType-instant.nc','/data/xlw/data_stream-oper_stepType-accum.nc',2)
model = Fuxi(input_channels=5).to(device).bfloat16()
criterion = L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.00025, betas=(0.9, 0.95), weight_decay=0.1,eps=1e-5)
epochs = 40000
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    current = 1
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        print("{}/{}".format(current,len(dataloader)))
        inputs, targets = inputs.to(device).bfloat16(), targets.to(device).bfloat16()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # print(targets.shape)  #[1, 5, 721, 1440]
        # print(outputs.shape)  #[1, 5, 721, 1440]

        loss.backward()
        optimizer.step()
        print(loss.item())
        running_loss += loss.item()
        current += 1
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')
    torch.save({'model': model.state_dict()}, 'model_param_{}.pth'.format(epoch + 1))
