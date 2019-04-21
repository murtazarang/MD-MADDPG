import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
# x = np.ones((3, 1))
# y = np.ones((4, 3))
#
# x = tf.expand_dims(x, axis=1)
#
# sess = tf.Session()
# print(sess.run(x * y))
#
# def key_model(input, scope, reuse=False):
#     with tf.variable_scope(scope, reuse=reuse):
#         out = layers.fully_connected(input, num_outputs=3, activation_fn=tf.nn.relu)
#     return out
#
# def network():
#     input = tf.placeholder(tf.float32, shape=(10, 3))
#
#     with tf.variable_scope("test1"):
#         x = key_model(input, "layer1")
#
#     with tf.variable_scope("test2"):
#         y = key_model(input, "layer1", reuse=True)
#
# network()
# for i in tf.get_default_graph().get_operations():
#     print(i.name)

# sess = tf.Session()
# # x = np.random.normal(loc=0, scale=1, size=(1, 3, 15))
# # y = np.ones((1, 3, 15))
# x =tf.placeholder(dtype=tf.float32, shape = (1, 3, 15))
# a = tf.transpose(x)
# print(a)
# z = np.ones((1, 128))
#
#
#
# import argparse
# parser=argparse.ArgumentParser()
# parser.add_argument('-auto', action='store_true', default=True)
# args=parser.parse_args()
# print(args)

print(os.path.dirname('/tmp/policy/'))