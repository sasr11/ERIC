from tqdm import tqdm
import sys
import time
import random
import itertools
import torch

def func_1():
    # 排列组合，包含（1,1），不同时包含（1,2）或（2,1）
    combinations = list(itertools.combinations_with_replacement(range(1, 11), 2))
    print(list(combinations[0]))
    print(type(list(combinations[0])))


if __name__ == "__main__":

    n, d = 2, 3
    tensor1 = torch.randn(n, d)
    tensor2 = torch.randn(n, d)
    print(tensor1)
    print("-------------------------")
    print(tensor2)
    print("+++++++++++++++++++++++++")

    # 扩展张量维度以计算两者之间的差
    tensor1_expanded = tensor1.unsqueeze(1)  # 扩展为 (n, 1, d)
    tensor2_expanded = tensor2.unsqueeze(0)  # 扩展为 (1, n, d)

    # 计算元素差的平方
    difference = tensor1_expanded - tensor2_expanded  # 形状为 (n, n, d)
    difference_squared = difference ** 2

    print(difference_squared)
    pass