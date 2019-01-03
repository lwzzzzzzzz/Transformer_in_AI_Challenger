# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import jieba
import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_valid_data, load_ch_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

# os.environ["CUDA_VISIBLE_DEVICES"]= '7'

def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Sources, Targets = load_valid_data()
    en2idx, idx2en = load_en_vocab()
    ch2idx, idx2ch = load_ch_vocab()

    #     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]

    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname + "_eval2", "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):

                    ### Get mini-batches 切片得到batch
                    x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        # 通过网络预测g.preds，feed_dict的g.y，是之前定义的全为0的preds
                        # 每次预测batch中所有句子的一个单词
                        # 因为multi-attention有各种mask存在，所以当预测y的第i个单词时，self-attentuon不会受后面单词的影响(seq-mask)
                        #                                       同时decoder-encoder-attention不会受0 <PAD>标记影响(query-mask)
                        # 所以可以一个一个单词训练。
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]

                    ### Write to file
                    # 通过zip把batch中的一个句子的source, target, pred取出来
                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        # " ".join获得整个句子，在</S>前的留下
                        got = " ".join(idx2ch[idx] for idx in pred).split("</S>")[0].strip()

                        fout.write("- source: " + source + "\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()

                        # bleu score
                        ref = jieba.lcut(target)
                        hypothesis = got.split()
                        print(hypotheses)
                        #总长小于3的句子不计算bleu，因为bleu对短的句子得分很高。
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)

                ## Calculate bleu score
                # list_of_refs的形状为  所有长度大于3的句子长度 * 1 * 该句句子长度
                # 没有batch的信息，因为batch只是一个训练参数
                # corpus_bleu第一个参数是  参考翻译句子的集合，可能有多个，第二个参数是机器翻译结果
                # 形状为  [ [[第一个参考翻译], [第一个参考翻译], [第一个参考翻译],...], [机器翻译结果] ]
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100 * score))

if __name__ == '__main__':
    eval()
    print("Done")

