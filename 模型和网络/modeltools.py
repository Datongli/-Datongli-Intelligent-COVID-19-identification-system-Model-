"""
该文件用于封装一些常用的模型训练函数，方便以后的调用
"""
import imageio.v2 as imageio
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import itertools
import os
import random
# import cv2
import matplotlib.pyplot as plt
import numpy as np
# PaddyDataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
# from pytorchtools import EarlyStopping
from tqdm import tqdm
import GhostNet
import efficientnet
import GhostNet_res
import Covnet
import Covnet_2
import Covnet_3
import ResNet


def labels_name(pre_list):
    """
    该函数用于定义使用的标签是什么
    :param pre_list: 标签的真实名称
    :return: 标签的字典
    """
    negative = pre_list[0]
    positive = pre_list[1]
    # 返回值，字典
    paddy_labels = {
        negative: 0,
        positive: 1
    }
    return paddy_labels


# 用于包装train和val数据的dataset迭代器，里面剔除了test数据
class PaddyDataSet_train_val(Dataset, ):
    def __init__(self, data_dir,  labels, test_rate,
                 transform=None, label_name=['negative', 'positive']):
        """
        数据集
        """
        self.label_name = {label_name[0]: 0, label_name[1]: 1}
        self.labels = labels
        self.test_rate = test_rate
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        # print(img.size)
        if img.size == self.temp.shape:
            img = img.resize((224, 224))
            # print(img.size)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # print(sub_dir)
                    label = self.labels[sub_dir]
                    data_info.append((path_img, int(label)))

        # data_info 里面包含了全部的数据以及对应的图片，从中选取 test_num 百分数的数据作为验证集
        data_num = len(data_info)  # 看一下数据集的长度 data_num
        data_num = int(self.test_rate * data_num)
        for i in range(data_num):
            selected_element = random.choice(data_info)
            test_data_info.append(selected_element)
            data_info.remove(selected_element)

        return data_info