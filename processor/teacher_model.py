from net.st_gcn import Model
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import time
import torchlight


# 从 weights_path 加载权重 - 该函数没有被调用过 - 待使用
def load_weights(model, weights_path, ignore_weights=None):
    # 无需考虑 ignore_weights !
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    # 从指定的 weights_path 加载权重
    print_log('Load weights from {}.'.format(weights_path))
    # 加载步骤 weights_path -> weights
    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],
                            v.cpu()] for k, v in weights.items()])

    # !-- filter weights 过滤 - 无ignore_weights无需考虑
    for i in ignore_weights:
        ignore_name = list()
        for w in weights:
            if w.find(i) == 0:
                ignore_name.append(w)
        for n in ignore_name:
            weights.pop(n)
            print_log('Filter [{}] remove weights [{}].'.format(i, n))

    # 打印已经加载的权重
    for w in weights:
        print_log('Load weights [{}].'.format(w))

    try:
        # 关键步骤 weight -> model
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        for d in diff:
            print_log('Can not find weights [{}].'.format(d))
        state.update(weights)
        model.load_state_dict(state)

    # 返回模型
    return model


def print_log(str, print_time=True):
    if print_time:
        str = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str
    if True:
        print(str)


def get_teacher_model(path='pre_trained/st_gcn.ntu-xsub.pt'):
    # 目标：加载教师模型
    graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    # 建立模型
    model = Model(in_channels=3, num_class=60, dropout=0.5, edge_importance_weighting=True, graph_args=graph_args)
    # 主程序
    model_t = load_weights(model, path)

    # gpu
    use_gpu = True
    # 单个Cuda, 这里是cuda 0
    device = [0]
    if use_gpu:
        gpus = torchlight.visible_gpu(device)
        torchlight.occupy_gpu(gpus)
        gpus = gpus
        dev = "cuda:0"
    else:
        dev = "cpu"

    # move modules to gpu
    model_t = model_t.to(dev)

    # model parallel
    if use_gpu and len(gpus) > 1:
        model_t = nn.DataParallel(model, device_ids=gpus)

    return model_t

# get_teacher_model(path='pre_trained/st_gcn.ntu-xsub.pt')