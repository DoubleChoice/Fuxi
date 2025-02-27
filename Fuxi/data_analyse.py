#import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import torch
from torch import parse_ir
from torch.utils.data import Dataset,DataLoader
import xarray as xr
import torch.nn as nn
import torch.optim as optim
from data.DataSet import WeatherDataset

# file = D:/ERA5/data_stream-oper_stepType-instant.nc
# #dict_keys(['number', 'valid_time', 'latitude', 'longitude', 'expver', 'u10', 'v10', 't2m', 'msl'])
# file = D:/ERA5/data_stream-oper_stepType-accum.nc
# dict_keys(['number', 'valid_time', 'latitude', 'longitude', 'expver', 'tp'])

list = ['u10', 'v10', 't2m', 'msl']

#A:/workspace/data/ERA5/data_stream-oper_stepType-instant.nc
#A:/workspace/data/ERA5/data_stream-oper_stepType-accum.nc


