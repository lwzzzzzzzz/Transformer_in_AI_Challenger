from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_ch_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '0,1,2,3'


class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            else:  # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

            # Load vocabulary
            en2idx, idx2en = load_en_vocab()
            ch2idx, idx2ch = load_ch_vocab()

            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(en2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")
                else:
                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=len(ch2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                                                  [tf.shape(self.decoder_inputs)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe")

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention")

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Final linear projection
            # 对最后一维做线性变换成词库这么长，对应每个单词的logits，然后将logits最大的索引记录下来，即预测值
            self.logits = tf.layers.dense(self.dec, len(ch2idx)) #(N, T, vocab_len)
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1)) # (N, T)
            # 把y中所有不是<PAD>出来的都由True转化为1.0
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            # acc表示的是  （一个batch中所有的非<PAD>的单词，预测对的数量求和）/（一个batch中所有的非<PAD>单词数量）
            # tips:tf.reduce_sum()未指定axis，即把所有维度都加起来
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            # 计算acc给summary监督学习过程。
            tf.summary.scalar('acc', self.acc)

            if is_training:
                # Loss
                # tf.one_hot(tensor, int),构造一个len(tensor)*int的tensor，tensor的值变成索引，对应位置为1.，其他为0.
                # 如果索引值大于int大小，则整行都是0.
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(ch2idx))) #y_smoothed因为one_hot变成了(N, T, vocab_len)
                # tf.nn.softmax_cross_entropy_with_logits实际上做的事情是：
                # 1.先对logits求softmax   2.再将vocab_len上的分布和y_label做交叉熵，得到一个(N, T)的向量
                # 即每一单词有一个loss
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed) # (N, T)
                # 将<PAD>出来的部分的loss去掉，再求mean_loss
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget)) #标量scale

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # Load vocabulary
    en2idx, idx2en = load_en_vocab()
    ch2idx, idx2ch = load_ch_vocab()

    # Construct graph
    g = Graph("train"); print("Graph loaded")
    # Start session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    ''' 正常的tf.train.string_input_producer()或tf.train.slice_input_producer()需要启动线程队列
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord) #启动QueueRunner，否则数据流图将一直挂起
    但是tf.train.Supervisor的managed_session会自动帮助你启动线程队列，并且管理线程，不需要try ... except ...
    '''
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # 一共跑num_epoch次epoch
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            # 对每次batch做一次train
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")