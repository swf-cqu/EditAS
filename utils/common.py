# encoding=utf-8

import re
import torch
import random
import importlib
import numpy as np
from typing import List, Iterable


PADDING = '<pad>'
CODE_PAD = '<pad>'
TGT_START = '<s>'
TGT_END = '</s>'
UNK = '<unk>'

ACTION_2_TGT_ACTION = {
    'insert': '<insert>',
    'delete': '<delete>',
    'replace': '<replace>',
    'equal': '<equal>'
}

FLOAT_TYPE = torch.float


def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # set random seed for all devices (both CPU and GPU)
    torch.manual_seed(seed)

    #torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    #torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    torch.backends.cudnn.deterministic = True # 采用默认算法
    torch.backends.cudnn.benchmark = False # 存在随机波动，因此将该设置关闭


def ids_to_input_tensor(word_ids: List[List[int]], pad_token: int, device: torch.device) -> torch.Tensor:
    sents_t = input_transpose(word_ids, pad_token)    # 一列代表一个数据
    sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
    return sents_var


def input_transpose(sents, pad_token):  # 转置padding
    max_len = max(len(s) for s in sents)  # 获取每个句子的长度
    batch_size = len(sents)  # 句子的个数

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])   # 句子对齐，拥有相同的长度

    return sents_t


def get_attr_by_name(class_name: str):
    class_tokens = class_name.split('.')
    assert len(class_tokens) > 1
    module_name = ".".join(class_tokens[:-1])
    # 动态导入包
    module = importlib.import_module(module_name)  # 一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    # 获取包下的 该属性值
    return getattr(module, class_tokens[-1])


def word_level_edit_distance(a: List[str], b: List[str]) -> int:  # 动态规划
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]


def recover_desc(sent: Iterable[str]) -> str:
    '''
    '''
    return re.sub(r' <con> ', "", " ".join(sent))

