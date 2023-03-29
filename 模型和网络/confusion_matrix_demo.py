"""
浅浅绘制一下新的混淆矩阵，使用大一些的字体
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


"""
浅浅绘制一下新的混淆矩阵，使用大一些的字体
"""
# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# 初始化混淆矩阵
cnf_matrix = np.zeros([2, 2])
n = 128 / 2
cnf_matrix[0][0] = 0.9 * n
cnf_matrix[0][1] = 0.1 * n
cnf_matrix[1][0] = 0.28 * n
cnf_matrix[1][1] = 0.72 * n
Confusion_matrix_path = os.path.join(r"C:\Users\ldt20\Desktop\图片", "all混淆矩阵" + nowTime)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          path=Confusion_matrix_path):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #         print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    #         print(cm)
    #     else:
    #         print('显示具体数字：')
    #         print(cm)
    plt.figure(dpi=320, figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 20})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontdict={'fontsize': 10})
    plt.yticks(tick_marks, classes, rotation=45, fontdict={'fontsize': 10})
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else '.0f'
    # fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "red",
                 fontdict={'fontsize': 40})

    plt.tight_layout()
    plt.xlabel('True label', fontdict={'fontsize': 20})
    plt.ylabel('Predicted label', fontdict={'fontsize': 20})
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.9)
    # plt.show()
    plt.savefig(path)


classes = ['negative', 'positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')
print('finish')