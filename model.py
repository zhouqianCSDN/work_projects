#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/29 
# @Author : zhouqian
# @File : model.py


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import jieba
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from simhash import Simhash
from text2vec import Word2Vec
from text2vec import Similarity


def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    D = np.zeros((m+1,n+1))
    # 初始化第一行和第一列
    for i in range(m+1):
        D[i][0] = i

    for j in range(n+1):
        D[0][j] = j

    for i in range(1,m+1):
        for j in range(1,n+1):
            left = D[i][j-1] + 1
            down = D[i-1][j] + 1

            left_down = D[i-1][j-1]
            if str1[i-1] != str2[j-1]:
                left_down += 1

            D[i][j] = min(left, down, left_down)

#     print(D[m][n])

    return 1 - D[m][n]/len(str1)


def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def simhash_similarity(text1, text2):
    """
    :param text1: 文本1
    :param text2: 文本2
    :return: 返回两篇文章的相似度
    """
    aa_simhash = Simhash(text1)
    bb_simhash = Simhash(text2)
    max_hashbit = max(len(bin(aa_simhash.value)), (len(bin(bb_simhash.value))))
    # 汉明距离
    distince = aa_simhash.distance(bb_simhash)
    similar = 1 - distince / max_hashbit
    return similar


def load_word2vec():
    w2v_model = Word2Vec("w2v-light-tencent-chinese")
    return w2v_model


def load_sentence_bert():
    t2v_model = Similarity("shibing624/text2vec-base-chinese")
    return t2v_model