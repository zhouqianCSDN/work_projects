#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/29 
# @Author : zhouqian
# @File : run.py

import argparse
import os
import time
from data_process import *
from model import *
from tool import init_logger, logger


def compute_similarity(str1, text_list):
    sim_list = []
    for i in text_list:
        a = tf_similarity(str1, i)
        b = simhash_similarity(str1, i)
#         c = edit_distance(str1, i)
#         final_sim = max(a,b)
        final_sim = a
        print(f'{str1}-----{i}的文本相似度：{final_sim}')
        sim_list.append(final_sim)
#     max_sim = max(sim_list)
#     if max_sim > 0.5:
#         max_index = sim_list.index(max_sim)
#         return text_list[max_index]
    return sim_list


def main(right_phrase, text, key_word):
    parser = argparse.ArgumentParser("Process......")
    parser.add_argument("--semantic_model", default="word2vec", type=str,
                        help="choice the model type for semantic, word2vec or Cosent, default is word2vec")

    parser.add_argument("--compute_logic", default="both", type=str,
                        help="the logic of compute similarity, default is both")

    parser.add_argument("--output_sorted_reference", default="machine learning", type=str,
                        help="the sorted_reference of final result, machine learning or deep learning, default is machine learning")
    args = parser.parse_args()
    path1 = os.path.dirname(__file__)
    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    model_name_or_path = os.path.join(path1, '/output')
    if not os.path.exists(model_name_or_path):
        os.makedirs(model_name_or_path, exist_ok=True)
    log_file = model_name_or_path + fr'/content_audit-{time_}.log'
    init_logger(log_file=fr'{log_file}')
    text_list = separate_text(right_phrase, text, key_word)
    sim_list = compute_similarity(right_phrase, text_list)
    start = time.time()
    if args.semantic_model == "word2vec":
        model = load_word2vec()
        t0 = time.time() - start
        logger.info('Load pretrained model:{},load model time:{:.2f}'.format(args.semantic_model, t0))
        sentence_embeddings_word2vec_ = model.encode(text_list, show_progress_bar=True)
        sentence_embeddings_word2vec = model.encode(["提高人民生活水平"], show_progress_bar=True)
        model_score = cos_sim(sentence_embeddings_word2vec, sentence_embeddings_word2vec_)
        t1 = time.time() - start
        logger.info("model calculate time: %.2f" % t1)
        model_score = model_score.reshape(-1)
    else:
        model = load_sentence_bert()
        t0 = time.time() - start
        logger.info('Load pretrained model:{},load model time:{:.2f}'.format(args.semantic_model, t0))
        model_score = model.get_scores([right_phrase], text_list)
        t1 = time.time() - start
        logger.info("model calculate time: %.2f" % t1)
        model_score = model_score.reshape(-1)
    total_score = np.array(sim_list) + model_score
    index1 = set(np.where(np.array(sim_list) > 0.4)[0])
    index2 = set(np.where(model_score > 0.7)[0])
    index3 = set(np.where(total_score > 1.3)[0])
    # print(index1)
    # print(index2)
    # print(index3)
    set_index = index1.intersection(index2, index3)
    if len(set_index) == 0:
        logger.info("%s ------在此文本段: %s 没有对应的错误表达" % (right_phrase, text))
    else:
        if args.output_sorted_reference == "machine learning":
            topN_index = findTopNindex(sim_list, 5)
            topN_score = [sim_list[i] for i in topN_index]
        else:
            topN_index = findTopNindex(model_score, 5)
            topN_score = [model_score[i] for i in topN_index]
        topN_text = [text_list[i] for i in topN_index]
        for sentence, score in zip(topN_text, topN_score):
            print("Sentence:", sentence)
            print("Score:", score)
    t2 = time.time() - start
    logger.info("total calculate time: %.2f" % t2)


if __name__ == "__main__":
    right_phrase = "提高人民生活水平"
    text = "坚持把保障和改善人民生活水平放在重要"
    key_word = "人民"
    main(right_phrase, text, key_word)









