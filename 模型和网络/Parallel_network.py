"""
此文件用于搭建两个并联的网络，最后使用一个全连接层合并
最后通过全连接层，生成两个节点
"""
"""
现在的策略是仿照中科院，在全连接层前加入了三层的卷积，然后最后是全连接
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
    def __init__(self, num_classes=2, dropout=0.2, include_top=True):
        super(parallel_net, self).__init__()
        # 属性分配
        self.dropout = dropout
        self.include_top = include_top
        self.conv1_outchannels = 128
        self.conv2_outchannels = 64

        # 部分一用于承接时频差分特性，暂定使用resnet网络
        # include_top=False 即代表不适用最后的全连接层，还是一个4维张量的输出形式
        self.part_1 = ResNet.resnet18(num_classes=32, include_top=False, dropout=self.dropout)
        # self.part_1 = Covnet_3.Covnet(drop_1=0.2, drop_2=0.2)
        # 部分二用于承接对数梅尔倒谱图，暂定使用resnet网络
        self.part_2 = ResNet.resnet18(num_classes=32, include_top=False, dropout=self.dropout)
        # self.part_2 = Covnet_3.Covnet(drop_1=0.2, drop_2=0.2)
        # self.part_2 = GhostNet.GhostNet(num_classes=256, dropout=0.2)
        # self.bn = nn.BatchNorm1d(256*2)

        self.conv1 = nn.Conv2d(in_channels=512 * 2, out_channels=self.conv1_outchannels,
                               kernel_size=3, stride=1, padding=2, bias=False)
        # 有bn层就不需要bias偏置
        self.bn1 = nn.BatchNorm2d(self.conv1_outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.conv1_outchannels, out_channels=self.conv2_outchannels,
                               kernel_size=3, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.conv2_outchannels)
        self.softmax = nn.Softmax(dim=1)

        # 自适应全局平均池化，无论输入特征图的shape是多少，输出特征图的(h,w)==(1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接分类
        self.fc_1 = nn.Linear(self.conv2_outchannels, num_classes)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc_2 = nn.Linear(128, 32)
        # self.fc_3 = nn.Linear(128, num_classes)

        # 卷积层权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
        # x = torch.tensor
        # 将通道连接起来
        x = torch.cat((x_1, x_2), dim=1)
        x = x.to(device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.include_top:
            # 全局平均池化
            x = self.avgpool(x)
            # 打平
            x = torch.flatten(x, 1)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            # 全连接分类
            x = self.fc_1(x)
            # x = self.relu(x)
            # x = self.fc_2(x)
            # x = self.relu(x)
            # x = self.fc_3(x)
            # 是不是应该加一层softmax
            # x = self.softmax(x)
        return x


class parallel_covnet(nn.Module):
    def __init__(self, num_classes=2, dropout_1=0.2, dropout_2=0.2, include_top=True):
        super(parallel_covnet, self).__init__()
        # 属性分配
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.include_top = include_top
        self.part_out = 64

        # 部分一用于承接时频差分特性，暂定使用resnet网络
        # include_top=False 即代表不适用最后的全连接层，还是一个4维张量的输出形式
        # self.part_1 = ResNet.resnet18(num_classes=32, include_top=False, dropout=self.dropout)
        self.part_1 = Covnet_3.Covnet(drop_1=self.dropout_1, drop_2=self.dropout_2, out=self.part_out)
        # self.part_1 = Covnet_3.Covnet(drop_1=0.2, drop_2=0.2)
        # 部分二用于承接对数梅尔倒谱图，暂定使用resnet网络
        # self.part_2 = ResNet.resnet18(num_classes=32, include_top=False, dropout=self.dropout)
        # self.part_1 = Covnet_3.Covnet(drop_1=self.dropout_1, drop_2=self.dropout_2)
        self.part_2 = Covnet_3.Covnet(drop_1=self.dropout_1, drop_2=self.dropout_2, out=self.part_out)
        # self.part_2 = Covnet_3.Covnet(drop_1=0.2, drop_2=0.2)
        # self.part_2 = GhostNet.GhostNet(num_classes=256, dropout=0.2)
        # self.bn = nn.BatchNorm1d(256*2)

        # 自适应全局平均池化，无论输入特征图的shape是多少，输出特征图的(h,w)==(1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(2 * self.part_out)
        # 全连接分类
        self.fc_1 = nn.Linear(2 * self.part_out, num_classes)

        # 卷积层权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
        # x = torch.tensor
        # 将通道连接起来
        x = torch.cat((x_1, x_2), dim=1)
        x = x.to(device)

        if self.include_top:
            # 全局平均池化
            # x = self.avgpool(x)
            # 打平
            # x = torch.flatten(x, 1)
            x = self.bn(x)
            # 全连接分类
            x = self.fc_1(x)
            # x = self.relu(x)
            # x = self.fc_2(x)
            # x = self.relu(x)
            # x = self.fc_3(x)
            # 是不是应该加一层softmax
            # x = self.softmax(x)
        return x