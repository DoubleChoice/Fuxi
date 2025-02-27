import gc

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import time


# file = D:/ERA5/data_stream-oper_stepType-instant.nc
# dict_keys(['number', 'valid_time', 'latitude', 'longitude', 'expver', 'u10', 'v10', 't2m', 'msl'])
# file = D:/ERA5/data_stream-oper_stepType-accum.nc
# dict_keys(['number', 'valid_time', 'latitude', 'longitude', 'expver', 'tp'])
class WeatherDataset(Dataset):
    def __init__(self, path1,path2 ,p = 2):
        self.X = torch.load(path1)
        self.Y = torch.load(path2)
        # print(self.X.shape) [N, 2, 721, 1440, 5]
        # print(self.Y.shape)   [N, 721, 1440, 5]
        # print(self.valid_time.shape)
        # print(self.lati.shape)
        # print(self.lon.shape)
        # print(self.u10.shape)
        # print(self.v10.shape)
        # print(self.t2m.shape)
        # print(self.tp.shape)
        # print(self.msl.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]
