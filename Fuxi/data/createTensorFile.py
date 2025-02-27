import gc
import numpy as np
import torch
import xarray as xr


# file = D:/ERA5/data_stream-oper_stepType-instant.nc
# dict_keys(['number', 'valid_time', 'latitude', 'longitude', 'expver', 'u10', 'v10', 't2m', 'msl'])
# file = D:/ERA5/data_stream-oper_stepType-accum.nc
# dict_keys(['number', 'valid_time', 'latitude', 'longitude', 'expver', 'tp'])

# /data/xlw/data_stream-oper_stepType-instant.nc
# /data/xlw/data_stream-oper_stepType-accum.nc

# A:/workspace/data/ERA5/data_stream-oper_stepType-instant.nc
# A:/workspace/data/ERA5/data_stream-oper_stepType-accum.nc

# 2022-01-01 00:00 ~ 2023-12-31 18:00 lead_time = 6
def create_timeseries(data, p,i):
    # [data_N, 721, 1440]
    for t in range(p+291*i, p+291*(i+1), p):
        if t == p+291*i:
            X = torch.unsqueeze(data[t - p:t], 0)
            y = torch.unsqueeze(data[t], 0)
        else:
            X = torch.cat((X, torch.unsqueeze(data[t - p:t], 0)), dim=0)
            y = torch.cat((y, torch.unsqueeze(data[t], 0)), dim=0)
    # X.shape [data_N - p, 2, 721, 1440]
    # y.shape [data_N - p, 721, 1440]
    # print(X.shape, y.shape)
    return X, y


def create_array(path1, path2, p=2, i=1):
    # 数据组织
    src = xr.open_dataset(path1)
    tp_file = xr.open_dataset(path2)
    # self.valid_time = src['valid_time'][:].values
    # self.lati = src['latitude'][:].values
    # self.lon = src['longitude'][:].values
    u10 = torch.tensor(src['u10'][:].values, dtype=torch.bfloat16)
    v10 = torch.tensor(src['v10'][:].values, dtype=torch.bfloat16)
    t2m = torch.tensor(src['t2m'][:].values, dtype=torch.bfloat16)
    tp = torch.tensor(tp_file['tp'][:].values, dtype=torch.bfloat16)
    msl = torch.tensor(src['msl'][:].values, dtype=torch.bfloat16)
    src.close()
    tp_file.close()
    X_u10, y_u10 = create_timeseries(u10, p,i)
    # print(X_u10.dtype)
    del u10
    X_v10, y_v10 = create_timeseries(v10, p,i)
    del v10
    X_t2m, y_t2m = create_timeseries(t2m, p,i)
    del t2m
    X_tp, y_tp = create_timeseries(tp, p,i)
    del tp
    X_msl, y_msl = create_timeseries(msl, p,i)
    del msl
    gc.collect()
    X = torch.stack([X_u10, X_v10, X_t2m, X_tp, X_msl], dim=2)     # [data_N - p, 2, 5, 721, 1440]
    torch.save(X, "D:/Data/X_p_{}.pt".format(i))
    del X
    del X_u10
    del X_v10
    del X_t2m
    del X_tp
    gc.collect()
    Y = torch.stack([y_u10, y_v10, y_t2m, y_tp, y_msl], dim=1)    # [data_N - p,5, 721, 1440]
    torch.save(Y, "D:/Data/Y_p_{}.pt".format(i))

    # print(valid_time.shape)
    # print(lati.shape)
    # print(lon.shape)
    # print(u10.shape)
    # print(v10.shape)
    # print(t2m.shape)
    # print(tp.shape)
    # print(msl.shape)

for i in range(1, 10):
    create_array('D:/Data/data_stream-oper_stepType-instant.nc', 'D:/Data/data_stream-oper_stepType-accum.nc', 2, i=i)
