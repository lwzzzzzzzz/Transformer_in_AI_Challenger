import tensorflow as tf
import numpy as np

def parse(x):
	return x+1
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
dataset = dataset.map(parse) # 让dataset中的每条数据都经过parse函数的解析
dataset=dataset.repeat().batch(3,drop_remainder=False)
# Dataset对数据进行处理的函数，返回仍是Dataset类
iterator = dataset.make_one_shot_iterator() # 构造迭代器
element = iterator.get_next() # get_next()迭代获取元素

with tf.Session() as sess:
	for i in range(5):
		print(sess.run(element))