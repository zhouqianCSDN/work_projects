#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/29 
# @Author : zhouqian
# @File : data_process.py


import numpy as np


def separate_text(s1, s2, key_word):
    text_list = []
    for i in range(len(s1) - 2, len(s1) + 3):
        for j in range(0, len(s2) - i + 1):
            if s2[j:j + i].find(key_word) == -1:
                continue
            text_list.append(s2[j:j + i])
            print(s2[j:j + i])

    print(len(text_list))

    return text_list


def findTopNindex(arr, N):
    return np.argsort(arr)[::-1][:N]


def cos_sim(arr1, arr2):
    norm1 = np.linalg.norm(arr1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(arr2, axis=-1, keepdims=True)
    arr1_norm = arr1 / norm1
    arr2_norm = arr2 / norm2
    model_sim = np.dot(arr1_norm, arr2_norm.T)
    return model_sim