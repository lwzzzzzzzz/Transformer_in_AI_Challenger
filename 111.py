import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '4'



a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
gpu_options = tf.GPUOptions(allow_growth=True)
sv = tf.train.Supervisor(logdir="./")
with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # 一共跑num_epoch次epoch
    for epoch in range(1, 100001):
        if sv.should_stop(): break
        # # 对每次batch做一次train
        # for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
        sess.run(c)