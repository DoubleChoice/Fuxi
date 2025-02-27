import torch.nn as nn
import torch
from src.models.Fuxi import Fuxi
import thop


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = Fuxi(input_channels=70,output_channels=70).to(device).bfloat16()
# [batch_size,p,721,1440,5]
data = torch.randn((1, 2,70, 721, 1440)).to(device).bfloat16()
output = model(data).bfloat16()
flops, params = thop.profile(model, inputs=(data,))
print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）