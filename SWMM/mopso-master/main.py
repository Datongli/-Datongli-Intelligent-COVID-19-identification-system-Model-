# encoding: utf-8
import numpy as np
from Mopso import *
import os
from pyswmm import Simulation, Nodes, Subcatchments, LidControls, LidGroup, LidUnit


def min_f(n):
    """
    粒子的最小值（n维）
    :param n: n维的问题（多少个LID）
    :return:
    """
    min_ = np.zeros(n)  # n个LID面积的最小值，全是零
    return min_


def max_f(csv_path, inp_path):
    """
    返回每一个集水区的面积，防止粒子优化时超出
    :param csv_path: csv文件的地址，里面有安装了LID的集水区的名称
    :param inp_path:inp文件的地址
    :return:一个ndarray数组，是每一个集水区的面积（平方米单位下）
    """
    # 读取inp文件，得到每一个集水区的面积最大值，作为每一个LID的面积上限
    # 注意：集水区的面积是公顷，需要换算成平方米
    max_list = []  # 定义一个空列表，用于承接集水区的面积，便于后续转化为ndarray数组
    name_list = []  # 定义含有集水区名称的空列表
    # 打开csv文件，获得每一个集水区的编号与面积（通过inp文件获得）
    csv_file = open(csv_path, "r")
    # 从csv文件中获得内容
    file_word = csv_file.read()
    csv_file.close()
    # 获得每一行的内容，集合为一个一维列表，里面元素是字符串
    lines = file_word.split('\n')[:-1]
    for line in lines:
        # 取出集水区的名称
        subcatchment_name = line.split(',')[0]
        name_list.append(subcatchment_name)
        # 打开inp文件，取出相应集水区的面积
        with Simulation(inp_path) as sim:
            subcatchments = Subcatchments(sim)
            S = subcatchments[subcatchment_name]
            # 取出集水区的面积，并换算成平方米
            max_list.append(S.area * 1e4)
    print(max_list)
    max_ = np.array(max_list)
    return max_, name_list


def main():
    w = 0.8  # 惯性因子
    c1 = 0.1  # 局部速度因子
    c2 = 0.1  # 全局速度因子
    particals = 100  # 粒子群的数量
    cycle_ = 30  # 迭代次数
    mesh_div = 10  # 网格等分数量
    thresh = 200  # 外部存档阀值
    LID_num = 53  # LID的个数
    # 存放着集水区与LID种类对应关系的csv文件的地址
    csv_path = r"D:\学习\竞赛\swmm\优化值集合\SUBCATCHMENTS-LID.csv"
    # inp文件的地址
    inp_path = r"D:\PythonProject\Image_recognition\SWMM\mopso-master\220629-zzmx.inp"
    """
    存放着集水区，LID平方米单价的字典
    """
    subcatchment_lid_price = {'S-10': '12.2505', 'S-11': '11.74095', 'S-12': '11.2314', 'S-13': '12.2505', 'S-14': '95.5009', 'S-15': '96.52', 'S-16': '109.7683', 'S-17': '12.2505', 'S-18': '11.74095', 'S-19': '47.46135', 'S-20': '96.52', 'S-21': '95.5009', 'S-23': '93.4627', 'S-24': '95.5009', 'S-26': '95.5009', 'S-27': '47.46135', 'S-28': '109.7683', 'S-29': '109.7683', 'S-30': '99.5773', 'S-31': '109.7683', 'S-33': '99.5773', 'S-34': '96.52', 'S-35': '109.7683', 'S-36': '96.52', 'S-41': '11.2314', 'S-42': '11.74095', 'S-43': '11.74095', 'S-44': '47.46135', 'S-45': '11.74095', 'S-46': '46.9518', 'S-47': '47.46135', 'S-48': '12.2505', 'S-49': '12.2505', 'S-50': '47.9709', 'S-51': '47.9709', 'S-52': '47.9709', 'S-53': '11.74095', 'S-54': '11.2314', 'S-55': '47.9709', 'S-56': '47.46135', 'S-61': '12.2505', 'S-62': '12.2505', 'S-63': '11.74095', 'S-64': '11.2314', 'S-65': '11.74095', 'S-66': '47.9709', 'S-67': '94.4818', 'S-68': '47.46135', 'S-70': '11.74095', 'S-71': '11.74095', 'S-9': '46.9518', 'S-92': '47.9709', 'S-93': '47.9709'}
    """
    需要换成我们的LID的个数
    """
    # n个LID的最小面积，全是零
    min_ = min_f(LID_num)
    # 粒子坐标的最大值与集水区的名称列表
    max_, name_list = max_f(csv_path=csv_path, inp_path=inp_path)
    """
    实例化多目标粒子群算法
    """
    mopso_ = Mopso(particals, w, c1, c2, max_, min_, thresh, inp_path, csv_path, name_list, subcatchment_lid_price, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)  # 经过cycle_轮迭代后，pareto边界粒子
    np.savetxt(os.path.join(r"D:\学习\竞赛\swmm\优化值集合", "pareto_in.txt"), pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt(os.path.join(r"D:\学习\竞赛\swmm\优化值集合", "pareto_fitness.txt"), pareto_fitness)  # 打印pareto边界粒子的适应值
    print("\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt")
    print("pareto边界的适应值保存于：/img_txt/pareto_fitness.txt")
    print("\n,迭代结束,over")


if __name__ == "__main__":
    main()
