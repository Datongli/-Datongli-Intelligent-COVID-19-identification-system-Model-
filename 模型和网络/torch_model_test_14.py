"""
同时搞清楚一下计算的过程
更改了写入txt的程序
此文件现在的问题是，模型训练后没有改进，仿佛没有调整，使得模型没有学习到东西
可以考虑从参数的初始化等方向入手
"""
"""
强烈建议检查一下one-hot编码的问题
解决，查看了一下官方示例，用法没有问题
官方示例是，input为3行5列的二维数组，labels为1行3列的数组，范围在0~4
这与我们现在的用法是一致的
"""
"""
现在的5折交叉验证的方式有问题
应该是在做交叉前，选择出来一部分的测试集，然后再在剩余的数据上进行5折交叉验证
已解决
"""
"""
此文件用于解决5折交叉验证有问题的情况
同时，使用 os.path.basename() 用法找出文件的名称
替换掉现在的rfind()用法
已解决
"""
"""
重新写一下读取数据的部分，现在是会有两个大文件夹，一个是训练加验证用的数据，一个是单独的测试用的数据
已完成
"""
import imageio.v2 as imageio
from torch.nn import init
import sys
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

# negative = 'cat'
# positive = 'dog'
negative = 'negative'
positive = 'positive'

# 工作目录
work_path = r"D:\学习\大创\data\训练数据集\model"
# 训练加验证数据集文件夹位置
filepath_train_val = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\测试集&训练集(2s)\训练集\chirplet"
# 测试数据集文件夹位置
filepath_test = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\测试集&训练集(2s)\测试集\chirplet"

paddy_labels = {negative: 0,
                positive: 1}


# 用于包装train和val数据的dataset迭代器，里面剔除了test数据
class PaddyDataSet_train_val(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        数据集
        """
        self.label_name = {negative: 0, positive: 1}
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
    def get_img_info(data_dir):
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
                    label = paddy_labels[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info


# 用于包装test数据的dataset迭代器
class PaddyDataSet_test(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        数据集
        """
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.test_info = self.get_img_info(data_dir)
        self.transform = transform
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img, label = self.test_info[index]
        img = Image.open(path_img).convert('RGB')
        # print(img.size)
        if img.size == self.temp.shape:
            img = img.resize((224, 224))
            # print(img.size)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.test_info)

    @staticmethod
    def get_img_info(data_dir):
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
                    label = paddy_labels[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info


def init_weights(layer):
    """
    参数初始化设置使用
    :param layer:
    :return:
    """
    # 如果为卷积层，使用 He initialization 方法正态分布生成值，生成随机数填充张量
    if type(layer) == nn.Conv2d:
        # nn.init.normal_(layer.weight, mean=0, std=1)
        nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


class Focal_Loss(torch.nn.Module):
    """
    二分类Focal Loss
    """

    def __init__(self, alpha=0.25, gamma=1):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps = 1e-7
        loss = []
        for i in range(len(labels)):
            loss_1 = -1 * (1 - self.alpha) * torch.pow((1 - preds[i][1]), self.gamma) * torch.log(preds[i][1] + eps) * \
                     labels[i]
            loss_0 = -1 * self.alpha * torch.pow(preds[i][1], self.gamma) * torch.log(1 - preds[i][1] + eps) * (
                    1 - labels[i])
            loss.append(loss_0 + loss_1)
        long = len(loss)
        loss = sum(loss)
        loss = torch.as_tensor(loss / long)
        # return torch.mean(loss)
        return loss


def getStat(all_data):
    '''
    用于计算自己（图片）数据集的均值与方差
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    train_loader = torch.utils.data.DataLoader(
        all_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print(type(train_loader))
    print(len(all_data))
    all_num = len(all_data)
    num = 0
    for X, _ in train_loader:
        num += 1
        print("共{}个，第{}个".format(all_num, num))
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(all_data))
    std.div_(len(all_data))
    return list(mean.numpy()), list(std.numpy())


# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
batch_size = 16  # 每个step训练batch_size张图片
epochs = 64  # 共训练epochs次
k = 5  # k折交叉验证
# 这两个是用于covnet的dropout参数
dropout_num_1 = 0.4
dropout_num_2 = 0.5
# 这个是用于resnet的dropout参数
resnet_dropout = 0.2
# 学习率
learning_rate = 1e-4
pre_score_k = []
labels_k = []
# wd：正则化惩罚的参数
wd = 0.01
print("wd:{}".format(wd))
# wd = None
# stop_epoch: 早停的批量数
stop_epoch = 10

# -------------------------------------------------- #
# （1）文件配置
# -------------------------------------------------- #
# 计算图片的总数量
negative_path = filepath_train_val + "\\" + negative
positive_path = filepath_train_val + "\\" + positive
all_photo_num = len(os.listdir(negative_path))
all_photo_num += len(os.listdir(positive_path))
# 计算两种样本的比例alpha = p:(n+p)
negative_num = len(os.listdir(negative_path))
positive_num = len(os.listdir(positive_path))
alpha = positive_num / (positive_num + negative_num)
print("alpha:{}".format(alpha))


# 显示一下文件夹的名称
dir_path = os.path.basename(filepath_train_val)
print(dir_path)
# 创建权重的文件夹
savepath = os.path.join(work_path, 'pth', dir_path)
cd = os.path.exists(savepath)
if cd:
    print("权重保存文件夹已存在")
else:
    print("创建权重保存文件夹")
    os.mkdir(savepath)
print(savepath)
# 判断保存图片文件夹是否存在
photo_folder = os.path.join(work_path, 'photo', dir_path)
cd = os.path.exists(photo_folder)
if cd:
    print("图片保存文件夹已存在")
else:
    print("创建图片保存文件夹")
    os.mkdir(photo_folder)

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')

# -------------------------------------------------- #
# （2）构造数据集
# -------------------------------------------------- #
# 计算数据集的均值与方差
transform = transforms.Compose([transforms.ToTensor()])
all_dataset = ImageFolder(root=filepath_train_val + '/', transform=transform)
image_mean, image_std = getStat(all_dataset)
print("image_mean:{}".format(image_mean))
print("image_std:{}".format(image_std))

# logmel 1:1
# image_mean = [0.327612, 0.5386462, 0.5382104]
# image_std = [0.36893702, 0.39973584, 0.32598126]

# TFDF logmel 1:1
# image_mean = [0.4685931, 0.9386316, 0.5017901]
# image_std = [0.2185139, 0.12783225, 0.21242227]

# TFDF
# image_mean = [0.46957287, 0.93973273, 0.50070703]
# image_std = [0.21676934, 0.126844, 0.21070491]

# chirplet
# image_mean = [0.010246435, 0.03795307, 0.59721166]
# image_std = [0.064348966, 0.14300026, 0.15483022]

# 读取数据集后再进行划分
data_dir = filepath_train_val
# 实例化一个对象，用于承接train和val的数据的迭代器
train_val_data = PaddyDataSet_train_val(data_dir=filepath_train_val,
                                        transform=transforms.Compose([
                                            # 将输入图像大小调整为224*224
                                            transforms.Resize((224, 224)),
                                            # # 数据增强，随机水平翻转
                                            # transforms.RandomHorizontalFlip(),
                                            # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=image_mean, std=image_std)
                                        ]))
# 实例化一个承接test数据的迭代器
test_data = PaddyDataSet_test(data_dir=filepath_test,
                              transform=transforms.Compose([
                                  # 将输入图像大小调整为224*224
                                  transforms.Resize((224, 224)),
                                  # # 数据增强，随机水平翻转
                                  # transforms.RandomHorizontalFlip(),
                                  # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=image_mean, std=image_std)
                              ]))

test_index = [i for i in range(len(test_data))]


# # 先划分成 5份
kf = KFold(n_splits=k, shuffle=True, random_state=34)
# 初始化混淆矩阵
cnf_matrix = np.zeros([2, 2])
# classes = data.classes


# -------------------------------------------------- #
# （3）加载模型
# -------------------------------------------------- #


# -------------------------------------------------- #
# （4）网络训练
# -------------------------------------------------- #


# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

"""
模型的训练
"""
k_num = 0
best_acc_all = 0
for train_index, val_index in kf.split(train_val_data):
    # 保存准确率最高的一次迭代
    best_acc = 0.0
    """
    每一折都要实例化新的模型，不然模型会学到测试集的东西
    """
    # # 2分类层
    net = ResNet.resnet18(num_classes=2, include_top=True, dropout=resnet_dropout)
    # net = models.resnet18(weights=None)
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 2)
    # net = Covnet_2.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = Covnet.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet.ghostnet()
    # net = Covnet_3.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2, out=2)
    # net = efficientnet.efficientnet_b0(num_classes=2)
    # net = GhostNet_res.resnet18()
    # net = ResNet_attention.resnet18(num_classes=1000, include_top=True)
    # 加载预训练权重
    # net.load_state_dict(torch.load(weightpath, map_location=device))
    # 为网络重写分类层
    # in_channel = net.fc.in_features  # 2048
    # net.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]
    # 给模型参数进行初始化
    # net.apply(init_weights)
    # 将模型搬运到GPU上
    net.to(device)
    # 获取网络名称
    net_name = net.__class__.__name__
    # 定义优化器
    # weight_decay：用于L2正则化，有助于抑制过拟合
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # 定义交叉熵损失
    # loss_function = Focal_Loss(alpha=alpha)
    """
    查一查reduction参数的含义
    """
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    # 写一个txt文件用于保存超参数
    file_name = r"{}\{}网络 {}.txt".format(photo_folder, net_name, nowTime)
    file = open(file_name, 'w', encoding='utf-8')
    if os.path.exists(file_name):
        file.write("batch_size:{}\n epoch:{}\n learning_rate:{}\n".format(batch_size, epochs, learning_rate))
        file.write("weight_decay:{}\n".format(wd))
        file.write(
            "dropout_1:{}, dropout_2:{}, resnet_dropout:{}\n".format(dropout_num_1, dropout_num_2, resnet_dropout))

    # 初始化一些空白矩阵
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    pre_score = []
    labels_epoch = []
    min_val_loss = 100
    # 显示此时是第k折交叉验证
    k_num += 1
    print("-" * 30)
    print("第{}折验证".format(k_num))
    train_fold = torch.utils.data.dataset.Subset(train_val_data, train_index)
    val_fold = torch.utils.data.dataset.Subset(train_val_data, val_index)
    # test_fold = torch.utils.data.dataset.Subset(test_data, test_index)
    # 计算训练集,验证集,测试集的大小
    train_num = len(train_fold)
    val_num = len(val_fold)
    # 打包成DataLoader类型 用于 训练
    train_loader = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # # 早停的实例
    # early_stopping = EarlyStopping(patience=4, delta=0.1, path=savepath + '/model_' + dir_path + "_第{}折验证{}层网络".format(k_num, net_num) + '.pth')

    """
    训练过程
    """
    for epoch in range(epochs):
        print('-' * 30, '\n', '共', epochs, '个epoch, 第', epoch + 1, '个epoch')
        file.write('{}\n,共{}个epoch,第{}个epoch\n'.format('-' * 30, epochs, epoch + 1))
        # 将模型设置为训练模型, dropout层和BN层只在训练时起作用
        net.train()
        # 计算训练一个epoch的总损失
        running_loss = 0.0
        epoch_acc = 0.0
        num_0 = 0


        # 每个step训练一个batch
        # enumerate：遍历，返回索引和元素
        # for step, data in tqdm(enumerate(train_loader), desc="train",unit='photo'):
        for step, data in tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout,
                               desc="共{}轮迭代，第{}轮迭代".format(epochs, epoch + 1), ncols=80, colour='yellow'):
            running_acc = 0.0
            num_0 += 1

            # data中包含图像及其对应的标签
            images, labels = data



            """
            尝试将数组反转成图片
            """
            # for i in images:
            #     transform = transforms.Compose([transforms.ToPILImage()])
            #     image = transform(i)
            #     plt.imshow(image)
            #     plt.show()
            #     i = i * torch.tensor(image_std).view(1, 3, 1, 1) + torch.tensor(image_mean).view(1, 3, 1, 1)
            #     transform = transforms.Compose([transforms.ToPILImage()])
            #     i = i[0]
            #     image = transform(i)
            #     plt.imshow(image)
            #     plt.show()

            # print(labels)
            # labels = torch.nn.functional.one_hot(labels, num_classes=2)
            # print(labels)

            # 梯度清零，因为每次计算梯度是一个累加
            optimizer.zero_grad()
            # 前向传播
            # output是torch.tensor类型的数据 [batch_size, 2]
            outputs = net(images.to(device))
            # print("output:{}".format(outputs))

            # 计算预测值和真实值的交叉熵损失,交叉熵损失也叫softmax损失
            loss = loss_function(outputs, labels.to(device))
            # loss = loss_function.forward(outputs, labels.to(device))

            # 计算acc
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]

            # 累加每个step的准确率
            running_acc = (predict_y == labels.to(device)).sum().item()
            epoch_acc += running_acc

            # # 梯度计算
            loss.backward()
            # 权重更新
            optimizer.step()

            # 累加每个step的损失
            running_loss += loss.item()

            # 打印每个step的损失和acc
            tqdm.write(f'loss:{loss} acc:{running_acc / batch_size}')

            # 查看每一步后的模型的参数更新
            # for name, param in net.named_parameters():
            #     # 名字 数据 梯度 是否需要梯度
            #     print(name, param.data, param.grad, param.requires_grad)
            #     print(np.shape(param.data))
            #     print(np.shape(param.grad))

            file.write("第{}折, step:{} loss:{} acc:{}\n".format(k_num,  step + 1, loss, running_acc / batch_size))

        # -------------------------------------------------- #
        # （5）网络验证
        # -------------------------------------------------- #
        net.eval()  # 切换为验证模型，BN和Dropout不进行参数的更新作用
        # net.train()  # 虽然是训练模式，但是因为没有写反向传播的代码，因此可以视为没有进行参数的更新

        acc = 0.0  # 验证集准确率
        val_loss_run = 0.0

        with torch.no_grad():  # 下面不进行梯度计算

            val_step = 0.0
            # 每次验证一个batch
            for data_val in val_loader:
                # 获取验证集的图片和标签
                val_images, val_labels = data_val
                # print(val_labels)
                # 前向传播
                outputs = net(val_images.to(device))

                """
                尝试反转成图片
                """
                # for i in val_images:
                #     transform = transforms.Compose([transforms.ToPILImage()])
                #     image = transform(i)
                #     plt.imshow(image)
                #     plt.show()
                #     i = i * torch.tensor(image_std).view(1, 3, 1, 1) + torch.tensor(image_mean).view(1, 3, 1, 1)
                #     transform = transforms.Compose([transforms.ToPILImage()])
                #     i = i[0]
                #     image = transform(i)
                #     plt.imshow(image)
                #     plt.show()

                # 计算预测值和真实值的交叉熵损失
                loss = loss_function(outputs, val_labels.to(device))
                # loss = loss_function.forward(outputs, val_labels.to(device))

                # 累加每个step的损失
                val_loss_run += loss.item()

                # 预测分数的最大值
                predict_y = torch.max(outputs, dim=1)[1]

                # 累加每个step的准确率
                acc += (predict_y == val_labels.to(device)).sum().item()

                val_step += 1

            # 计算所有图片的平均准确率
            acc_val = acc / val_num
            acc_train = epoch_acc / train_num

            # 打印每个epoch的训练损失和验证准确率
            print(f'total_train_loss:{running_loss / (step + 1)}, total_train_acc:{acc_train}')
            print(f'total_val_loss:{val_loss_run / val_step}, total_val_acc:{acc_val}')
            train_loss.append(running_loss / (step + 1))
            train_acc.append(acc_train)
            val_loss.append(val_loss_run / val_step)
            val_acc.append(acc_val)
            file.write('total_train_loss:{}, total_train_acc:{}\n'.format(running_loss / (step + 1), acc_train))
            file.write('total_val_loss:{}, total_val_acc:{}\n'.format(val_loss_run / val_step, acc_val))

            # 进行早停的检查
            if val_loss[-1] <= min_val_loss:
                min_val_loss = val_loss[-1]
                epoch_num = epoch + 1
            if val_loss[-1] >= min_val_loss + 0.2:
                if (epoch + 1) - epoch_num >= stop_epoch:
                    # 保存的权重名称
                    savename = savepath + '\\model_' + dir_path + "_第{}折验证".format(
                        k_num) + net_name + "网络" + '.pth'
                    # 保存当前权重
                    torch.save(net.state_dict(), savename, _use_new_zipfile_serialization=False)
                    break

            # -------------------------------------------------- #
            # （6）权重保存
            # -------------------------------------------------- #
            # 保存每一折验证的最好权重
            if acc_val > best_acc:
                # 更新最佳的准确率
                best_acc = acc_val
                # 保存的权重名称
                savename = savepath + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
                # 保存当前权重
                torch.save(net.state_dict(), savename, _use_new_zipfile_serialization=False)

            # 保存整个训练中的最好权重
            if acc_val > best_acc_all:
                # 更新最佳的准确率
                best_acc_all = acc_val
                # 保存的权重名称
                savename = savepath + '\\model_' + dir_path + "最好的权重" + net_name + "网络" + '.pth'
                # 保存当前权重
                torch.save(net.state_dict(), savename, _use_new_zipfile_serialization=False)


        # 学习率更新：根据回带的次数来更新学习率
        scheduler.step()

    """
    测试集，用于判断测试的准确率以及绘制roc曲线和混淆矩阵
    """
    # 使用刚刚训练的权重
    savename = savepath + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
    weightpath = savename
    # 初始化网络
    net = ResNet.resnet18(num_classes=2, include_top=True)
    # net = models.resnet18(weights=None)
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 2)
    # net = Covnet_2.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet.ghostnet()
    # net = Covnet_3.Covnet()
    # net = efficientnet.efficientnet_b0(num_classes=2)
    # net = Covnet.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet_res.resnet18()
    # 为网络重写分类层
    # in_channel = net.fc.in_features  # 2048
    # net.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]
    # 加载权重
    net.load_state_dict(torch.load(weightpath, map_location=device))
    # 模型切换成验证模式，目的是让dropout和bn切换形式
    net.eval()
    # 将模型搬运到GPU上
    net.to(device)
    test_acc = 0.0
    pre_score = []
    labels_epoch = []
    """
    这是是模型的测试过程
    """
    with torch.no_grad():
        test_step = 0.0
        for data_test in test_loader:
            # 获取测试集的图片和标签
            test_images, test_labels = data_test
            #  前向传播
            outputs = net(test_images.to(device))
            # 添加softmax层
            # outputs = nn.Softmax(dim=1)(outputs)
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]
            # 累加每个step的准确率
            test_acc += (predict_y == test_labels.to(device)).sum().item()
            test_step += 1

            # 准备roc曲线所需要的数据
            positive_pre = outputs[:, 1]
            positive_pre = positive_pre.cpu()
            positive_pre = positive_pre.detach().numpy()
            positive_pre = positive_pre.tolist()
            labels = test_labels.detach().numpy()
            labels = labels.tolist()
            pre_score += positive_pre
            labels_epoch += labels

            # 更新混淆矩阵
            for index in range(len(test_labels)):
                cnf_matrix[labels[index]][predict_y[index]] += 1

        test_file_num = batch_size * len(test_loader)
        # 计算测试集图片的平均准确率
        acc_test = test_acc / test_file_num
        # 打印测试集的准确率
        print("第{}折测试集的acc：{}".format(k_num, acc_test))
        file.write("第{}折测试集的acc：{}\n".format(k_num, acc_test))

    # 保存k折的roc参数
    pre_score_k.append(pre_score)
    labels_k.append(labels_epoch)

    # 每一折验证的时候，都绘制loss和acc曲线
    # 加时间戳
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    # plt.ylim((0, 1))
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(photo_folder + "\\" + net_name + "网络 model_loss_第{}折_".format(k_num) + str(nowTime) + ".jpg")
    # plt.show()
    # plt.xlim((0,50))
    # plt.ylim((0,1))
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    # plt.ylim((0, 1))  # 限制一下绘图的幅度，更具有代表性一些
    plt.legend(["train", "val"], loc="lower right")
    plt.savefig(photo_folder + "\\" + net_name + "网络 model_acc_第{}折_".format(k_num) + str(nowTime) + ".jpg")

"""
k折交叉验证的话，在前面绘制了loss和acc
绘制混淆矩阵以及每一折的ROC曲线并取平均，计算每一折AUC并取平均
"""
file.close()
# 以下是用于绘制ROC曲线的代码部分
# # 以下是用于绘制ROC曲线的代码部分
avg_x = []
avg_y = []
sum = 0
clr_1 = 'tab:green'
clr_2 = 'tab:green'
clr_3 = 'k'

plt.figure()
for i in range(k):
    fpr, tpr, thersholds = roc_curve(labels_k[i], pre_score_k[i])
    avg_x.append(sorted(random.sample(list(fpr), len(list(fpr)))))
    avg_y.append(sorted(random.sample(list(tpr), len(list(tpr)))))
    roc_auc1 = auc(fpr, tpr)

    roc_auc = roc_auc1 * 100
    sum = sum + roc_auc
    plt.plot(fpr, tpr, label='V-' + str(i + 1) + ' (auc = {0:.2f})'.format(roc_auc), c=clr_1, alpha=0.2)

data_x = np.array(avg_x, dtype=object)
data_y = np.array(avg_y, dtype=object)
avg = sum / k

# 准备数据
data_x_plt = []

data_x_num = len(data_x[0])
if data_x_num >= len(data_x[1]):
    data_x_num = len(data_x[1])
if data_x_num >= len(data_x[2]):
    data_x_num = len(data_x[2])
if data_x_num >= len(data_x[3]):
    data_x_num = len(data_x[3])
if data_x_num >= len(data_x[4]):
    data_x_num = len(data_x[4])

for i in range(5):
    data_x[i] = sorted(random.sample(data_x[i], data_x_num))

for i in range(data_x_num):
    a = 0.0
    a += data_x[0][i]
    a += data_x[1][i]
    a += data_x[2][i]
    a += data_x[3][i]
    a += data_x[4][i]
    a = a / k
    data_x_plt.append(a)

data_y_plt = []
data_y_num = len(data_y[0])
if data_y_num >= len(data_y[1]):
    data_y_num = len(data_y[1])
if data_y_num >= len(data_y[2]):
    data_y_num = len(data_y[2])
if data_y_num >= len(data_y[3]):
    data_y_num = len(data_y[3])
if data_y_num >= len(data_y[4]):
    data_y_num = len(data_y[4])

for i in range(5):
    data_y[i] = sorted(random.sample(data_y[i], data_y_num))

for i in range(data_y_num):
    a = 0.0
    a += data_y[0][i]
    a += data_y[1][i]
    a += data_y[2][i]
    a += data_y[3][i]
    a += data_y[4][i]
    a = a / k
    data_y_plt.append(a)

plt.plot(data_x_plt, data_y_plt, label='AVG (auc = {0:.2f})'.format(avg), c=clr_2, alpha=1, linewidth=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
plt.legend(loc='lower right', frameon=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(photo_folder + "\\" + net_name + "网络 model_ROC_" + str(nowTime) + ".jpg")
plt.show()

"""
绘制混淆矩阵，并保存
"""
Confusion_matrix_path = photo_folder + "\\" + net_name + "网络 Confusion matrix" + str(nowTime) + ".jpg"


# 绘制混淆矩阵
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
    plt.xlabel('Predicted label', fontdict={'fontsize': 20})
    plt.ylabel('True label', fontdict={'fontsize': 20})
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.9)
    # plt.show()
    plt.savefig(path)


# 第一种情况：显示百分比
# classes = ['cat', 'dog']
classes = ['negative', 'positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')

# # 第二种情况：显示数字
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')
