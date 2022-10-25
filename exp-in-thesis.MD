# 遮挡骨架识别
> 仅供在忘记具体调整位置时的参考
## 训练和测试参数
1. ./config/st_gcn/ntu_xsub/*.yaml
2. ./config/st_gcn/ntu_xview/*.yaml

## GPU相关参数
./processor/io.py, line 60

## 训练/测试在Epoch上的步骤
./processor/processor.py 
### 训练
line 110
1. 输出准确率
2. 模型保存
3. 训练、测试的参数

### 测试
line 146
1. 输出准确率
2. 保存测试结果

## 训练/测试在Batch上的步骤
./processor/recognition.py
1. Loss函数的定义
2. 前向、反向传播过程
3. 准确率等数据汇总统计
> Waiting 进一步修改，完成Loss函数的设计（包括tau等超参数）

## 导入教师模型
./processor/teacher_model.py
- 从*.pt格式的文件导入，返回model，该函数被recognition.py调用

## 处理NTU骨架数据，生成遮挡
./tools/utils/gen_occlusion.py
> Waiting 进一步修改，完善批量、自动的处理过程

## 模型保存、模型加载的具体实现 
./torchlight/torchlight/io.py
- 主要用于借鉴其中的处理思路，目前不需要关注

## 预处理骨架数据
./tools/ntu_gendata.py
```shell
python tools/ntu_gendata.py --data_path 此处输入不以斜杠开头的目录
```

## TensorBoard
```shell
tensorboard --logdir runs
```

## 启动训练和测试
1. 训练
python main.py recognition -c config/st_gcn/<dataset>/train.yaml [--work_dir <work folder>]

2. 测试 
python main.py recognition -c config/st_gcn/<dataset>/test.yaml --weights <path to model weights>
python main.py recognition -c config/st_gcn/ntu-xview/test.yaml

## 3月31日修改
在训练学生网络时，学生网络只使用缺失骨架进行训练
但教师网络应当输入对应的完整骨架，而不是缺失的骨架
1. train.yaml 的扩增

## GPU参数位置
1. Student Model的GPU参数位置
processor/io.py
line 70: cuda:0
2. Teacher Model的GPU参数位置
processor/teacher_model.py
line 77: cuda:0