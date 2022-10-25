# load config file
import torch
import yaml

f = open('./config/st_gcn/ntu-xsub/train.yaml', 'r')
default_arg = yaml.load(f, Loader=yaml.FullLoader)
# 由此可以得出 graph_args 是一个字典
# print(default_arg)


def func(layout='openpose', strategy='uniform', max_hop=1, dilation=1):
    print(layout)
    print(strategy)


def func2(in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
    print(kwargs)


#func(**graph_args)
#func2(in_channels=3, num_class=60, dropout=0.5, edge_importance_weighting=True, graph_args=graph_args)
# 上述代码结论：
# 传入的graph_args是一个字典
# 多余的参数会被收集到kwargs

import torch.nn.functional as F

T = 1

a = torch.Tensor([[0.4, 0.8, 0.9, 0.8, 0.2]])
b = torch.Tensor([[0.4, 0.8, 0.9, 0.8, 0.5]])

a = F.log_softmax(a/T, dim=1)
b = F.softmax(b/T, dim=1)

# print(a, b)

distillation_loss = F.kl_div(a,b,reduction='sum')

# print(distillation_loss)

import numpy as np
data = np.load("data/NTU-RGB-D/complete/xsub/train_data.npy", mmap_mode='r')
print(data)



















