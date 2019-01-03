from __future__ import print_function
from test.hypers import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
import nltk
import jieba


def load_ch_vocab():
    vocab = []
    with codecs.open('preprocessed/ch.vocab.tsv', 'r', 'utf-8') as f:
        for line in f.readlines():
            if int(line.split("\t")[1]) >= hp.min_cnt:
                vocab.append(line.split("\t")[0])
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    en2idx, idx2en = load_en_vocab()
    ch2idx, idx2ch = load_ch_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # 当word在字典时，字典get函数返回1，即对应<UNK>的idx
        x = [en2idx.get(word, 1) for word in nltk.word_tokenize(source_sent)+["</S>"]]  # 1: OOV, </S>: End of Text
        y = [ch2idx.get(word, 1) for word in jieba.lcut(target_sent)+["</S>"]]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        # perpro中的<PAD>的idx是0，所以pad进去的int也是0，这里把所有不足hp.maxlen的x,y都在后面补0，使之一样长。
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data():
    # 将decoder和encoder的train数据处理成一条一条的句子的list，即de_sents和en_sents，将这两个送给create_data处理
    # 并且这些句子因为if语句，把 <开头的标签 和 单独的/n行 都去掉后，在进行regex匹配。
    # 题外话：当打印de_sents[-1]不是hp.source_train最后一句的原因是，多线程读取源文件，最后一个读到的不一定是最后一行。
    # train_seg = []
    # for root, dirs, files in os.walk(hp.source_target_train_dir):
    #     for file in files:
    #         train_seg.append(os.path.join(root, file))
    ch_sents, en_sents = [], []
    # for each_seg in train_seg:
    with codecs.open("./corpora/tt.txt", 'r') as f:

        for line in f.readlines():
            en_sents.append(line.split("\t")[2])
            ch_sents.append(line.split("\t")[3].replace("\n",""))
            print(ch_sents)
        print(ch_sents)
    X, Y, Sources, Targets = create_data(en_sents, ch_sents)
    return X, Y


def load_test_data():
    def _refine(line):
        # 把所有的<>内容都清除掉
        line = regex.sub("<[^>]+>", "", line)
        # 同之前的处理，把所有的非space /t /n 和 拉丁文 和 ' 符号都一律用""代替
        line = regex.sub("[^\s\p{Latin}']", "", line)
        # 移除line首尾的空格（default）
        return line.strip()

    # _refine每次处理一行
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets  # (1064, 150)


def get_batch_data():
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size
    # X Y分别是train的en和de的向量化形式，不够长的已经补0，所以打印查看后面都是0
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # 从X Y到x y看似没什么变化，只是将原数据组织成batch形式
    # 但是用tf函数实现，多线程技术，让程序shuffle_batch效率高的不止一点

    # Create Queues
    # 当num_epoch为None时，tf.train.slice_input_producer可以无限迭代得往队列里塞所有数据
    # 很多时候，我们时采用在train.py文件中，通过for循环，使其跳出slice_input_producer操作，来控制epoch次数，而不是限制队列的迭代次数

    # input_queues里数据得形式是X和Y每一对应行的zip    形式如 [X[0],Y[0]]的list 在队列里。
    # 其功能和名字也很像， X,Y同一维度切片后list   slice_
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    # tf.train.shuffle_batch干的事是，把input_queues里的东西，用多线程方法，按batch_size个打包起来
    # 最后不足batch_size个得被舍弃，这就是为什么 num_batch=len(X)//hp.batch_size 是//地板除得原因
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size,
                                  capacity=hp.batch_size * 64,
                                  min_after_dequeue=hp.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()
