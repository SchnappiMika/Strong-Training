import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import time 
import os
import sys

def pair_distance(x: Tensor, y: Tensor) -> Tensor:
    """计算 x 和 y 之间的成对欧氏距离"""
    # x: (batch, dim), y: (num_actions, dim)
    # 返回: (batch, num_actions)
    return torch.cdist(x, y, p=2)


def weight_init(m: nn.Module):
    """自定义权重初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec

    def __init__(self, experiment_name, seed, log_path="experiments", restore=False, method=("file", "term")):
        """
        experiment_name: 实验名（参数组合字符串）
        seed: 随机种子（文件夹名）
        log_path: 根目录
        restore: 是否追加
        method: ("file", "term") 支持多种输出方式
        """
        self.terminal = sys.stdout
        self.file = "file" in method
        self.term = "term" in method
        self.temp = time.time()

        # 自动创建路径
        self.log_dir = os.path.join(log_path, experiment_name, str(seed))
        os.makedirs(self.log_dir, exist_ok=True)
        log_file_path = os.path.join(self.log_dir, "logfile.log")

        if self.file:
            if restore and os.path.exists(log_file_path):
                self.log = open(log_file_path, "a")
            else:
                self.log = open(log_file_path, "w")

    def write(self, message):
        if self.term:
            self.terminal.write(message)
            self.terminal.flush()

        if self.file:
            self.log.write(message)
            if (time.time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time.time()

    def flush(self):
        if self.file:
            self.log.flush()
            os.fsync(self.log.fileno())

    def close(self):
        if self.file:
            self.log.close()
