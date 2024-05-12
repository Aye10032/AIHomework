#!/usr/bin/env python
###################################################################
# File Name: data_gen.py
# Author: Li Zhen
# Email: lizhen2014@ict.ac.cn
# Created Time: 2020-02-13 09:24:58 CTS
# Description: 
###################################################################
import random as rd
from typing import TextIO


def num2str(num: int, mode: int, width: int):
    """
    将数字转换为字符串形式，根据模式选择二进制或十六进制表示，确保结果字符串的长度至少为指定宽度。

    :param num: 需要转换的数字
    :param mode: 转换模式，0代表二进制，非0代表十六进制
    :param width: 结果字符串的最小长度
    :return: 转换后的字符串，长度至少为width
    """

    # 根据模式选择转换为二进制或十六进制字符串
    if mode == 0:
        str_ori = bin(num)[2:]
    else:
        str_ori = hex(num)[2:]

    # 确保结果字符串的长度至少为width
    if len(str_ori) < width:
        text = (width - len(str_ori)) * "0" + str_ori
    else:
        text = str_ori

    return text


def verctor_gen(pfr_io: TextIO, nfr_io: TextIO, sfr_io: TextIO, iter_num: int):
    """
    生成向量并进行特定计算，然后将计算结果写入文件。

    :param pfr_io: 写入部分和结果的文件对象
    :param nfr_io: 写入神经元结果的文件对象
    :param sfr_io: 写入同步结果的文件对象
    :param iter_num: 迭代次数
    :return: 无返回值
    """

    line_iter = 32  # 每行迭代次数
    base = 2 ** 12  # 基数
    width = 4  # 数字宽度

    partsum = 0  # 部分和初始化
    # 主迭代过程
    for i in range(iter_num):
        neu_str = ""
        syn_str = ""

        # 生成神经元和同步字符串
        for k in range(line_iter):
            neu = rd.randint(0, 2 ** 12) % base
            syn = rd.randint(0, 2 ** 12) % base
            neu_str = neu_str + num2str(neu, 1, width)
            syn_str = syn_str + num2str(syn, 1, width)

            # 调整数值确保在基数的一半以下
            if neu >= (base / 2):
                neu = neu - base
            if syn >= (base / 2):
                syn = syn - base
            # 计算部分和
            partsum += neu * syn
        # 将神经元和同步结果写入文件
        nfr_io.write(neu_str + "\n")
        sfr_io.write(syn_str + "\n")
    # 打印部分和
    print("======================")
    print(partsum)
    print("======================")

    # 调整部分和确保非负，然后写入文件
    if partsum < 0:
        partsum += 2 ** 32

    pfr_io.write(num2str(partsum, 0, 32) + "\n")


# 主程序入口
if __name__ == "__main__":
    # 打开三个文件，分别用于记录神经元信息、权重信息和结果信息
    nfr = open("neuron_my", "w+")  # 神经元文件，写入模式
    sfr = open("weight_my", "w+")  # 权重文件，写入模式
    pfr = open("result_my", "w+")  # 结果文件，写入模式

    rd.seed(1)
    # 调用verctor_gen函数多次，分别以不同的参数进行计算并记录结果
    verctor_gen(pfr, nfr, sfr, 0x14)  # 以20为参数调用
    verctor_gen(pfr, nfr, sfr, 0x1e)  # 以30为参数调用
    verctor_gen(pfr, nfr, sfr, 0x28)  # 以40为参数调用
    verctor_gen(pfr, nfr, sfr, 0x32)  # 以50为参数调用
