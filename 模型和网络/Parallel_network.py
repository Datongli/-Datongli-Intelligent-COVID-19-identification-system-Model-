"""
此文件用于搭建两个并联的网络，最后使用一个全连接层合并
最后通过全连接层，生成两个节点
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import ResNet
import Covnet_3
import GhostNet

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

"""
新建一个类，用于表示并行网络
"""
class parallel_net(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(parallel_net, self).__init__()
        # 属性分配
        self.dropout = dropout

        # 部分一用于承接时频差分特性，暂定使用resnet网络，最后两个节点
        self.part_1 = ResNet.resnet18(num_classes=32, include_top=True, dropout=self.dropout)
        # self.part_1 = Covnet_3.Covnet(drop_1=0.2, drop_2=0.2)
        # 部分二用于承接对数梅尔倒谱图，暂定使用resnet网络，最后两个节点
        self.part_2 = ResNet.resnet18(num_classes=32, include_top=True, dropout=self.dropout)
        # self.part_2 = Covnet_3.Covnet(drop_1=0.2, drop_2=0.2)
        # self.part_2 = GhostNet.GhostNet(num_classes=256, dropout=0.2)
        # self.bn = nn.BatchNorm1d(256*2)

        self.softmax = nn.Softmax(dim=1)
        # 全连接分类
        self.fc_1 = nn.Linear(32*2, num_classes)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc_2 = nn.Linear(128, 32)
        # self.fc_3 = nn.Linear(128, num_classes)




    # 前向传播
    def forward(self, data_1, data_2):
        x_1 = self.part_1(data_1)
        x_2 = self.part_2(data_2)
        # x_1 = x_1.cpu()
        # x_2 = x_2.cpu()
        # x_1 = x_1.detach().numpy().tolist()
        # x_2 = x_2.detach().numpy().tolist()
        # for i in range(len(x_1)):
        #     for j in range(len(x_2[0])):
        #         x_1[i].append(x_2[i][j])
        # x = torch.tensor(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = x.to(device)
        # x = self.bn(x)
        # 全连接分类
        x = self.fc_1(x)
        # x = self.relu(x)
        # x = self.fc_2(x)
        # x = self.relu(x)
        # x = self.fc_3(x)
        # 是不是应该加一层softmax
        # x = self.softmax(x)
        return x



