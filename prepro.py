from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import jieba
import codecs
import os
from nltk.tokenize import word_tokenize
from collections import Counter


def make_vocab(word2cnt, fname):

    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        # 先将前四个特殊字符放在前面
        fout.write(
            "{}\t10000000000\n{}\t10000000000\n{}\t10000000000\n{}\t10000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        # most_common指定参数时，按counter数排序
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    word2cnt_ch = Counter()
    word2cnt_en = Counter()
    train_seg = []
    for root, dirs, files in os.walk(hp.source_target_train_dir):
        for file in files:
            train_seg.append(os.path.join(root, file))

    for each_seg in train_seg:
        with codecs.open(each_seg, 'r', 'utf-8') as f:
            i = 1
            word2cnt_ch_seg = Counter()
            word2cnt_en_seg = Counter()
            for line in f.readlines():
                i += 1
                line_en = line.split("\t")[2]
                line_ch = line.split("\t")[3].rstrip("\n")
                ch_list = jieba.lcut(line_ch, cut_all=False, HMM=True)
                word2cnt_ch_seg.update(ch_list)
                en_list = word_tokenize(line_en)
                word2cnt_en_seg.update(en_list)
                if i%100000==0:
                    print(i)
            word2cnt_ch.update(word2cnt_ch_seg)
            word2cnt_en.update(word2cnt_en_seg)

            print("one seg done")
    make_vocab(word2cnt_ch, "ch.vocab.tsv")
    make_vocab(word2cnt_en, "en.vocab.tsv")
    print("Done")