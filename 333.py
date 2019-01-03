import tensorflow as tf
a,b = [5, 7]
position_ind = tf.tile(tf.expand_dims(tf.range(a), 0), [b, 1])
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(position_ind))
