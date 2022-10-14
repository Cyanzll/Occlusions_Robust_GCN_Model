import numpy as np
import os

# 读取骨架数据

def read_skeleton(file):
    # 打开骨架文件 -> f
    with open(file, 'r') as f:
        # 字典：存放骨架序列
        skeleton_sequence = {}
        # 第一行：读取帧数
        skeleton_sequence['numFrame'] = int(f.readline())
        # 骨架序列列表：帧的信息
        skeleton_sequence['frameInfo'] = []
        # 按照帧数进行读取
        for t in range(skeleton_sequence['numFrame']): # 如：0 - 102
            frame_info = {}
            # 读行：人物编号
            frame_info['numBody'] = int(f.readline())
            # 列表：身体信息 按照帧来做记录
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']): # 如：0
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                # split() 按空格分割 -> 列表
                body_info = {
                    k: float(v)
                    # zip 对象中包含 (k, v) 元组
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                # 对于一个帧的关节信息，是一个列表的形式
                body_info['jointInfo'] = []
                # 按照关节点 joint 读取坐标等信息
                for v in range(body_info['numJoint']): # 如：从0 - 24
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    # 读完一个关节，追加到body_info列表中
                    body_info['jointInfo'].append(joint_info)
                # 读完一个帧，追加到frame_info列表中 注意，frame_info本身就是一个列表
                frame_info['bodyInfo'].append(body_info)

            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

# 下一步工作重点：
# 根据骨架数据 读取文件名
# 参考这种方式，可以修改得到遮挡数据集的生成程序
def read_xyz(file, max_body=2, num_joint=25):
    # 直接执行上面的函数
    seq_info = read_skeleton(file)
    # (3, 103, 25, 2) -> C T V M
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    # print(data)
    return data

# read_xyz("../../dataset/ntu-rgb-d/S001C001P001R001A001.skeleton")
# read_xyz('./occlude/S001C001P001R001A001.skeleton')