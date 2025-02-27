from data.DataSet import WeatherDataset
from src.models.modules.L1_loss import L1Loss
import torch
import xarray as xr
import numpy as np


xr.open_dataset('data_stream-oper_stepType-accum.nc')