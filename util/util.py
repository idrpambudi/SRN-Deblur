import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

"""
Implementation from:
https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
"""
def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                        initializer=tf.constant_initializer(0.1),
                        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def res_bottleneck_dsconv(x, dim, ksize, stride=1, scope='rb_dwconv', activation_fn=parametric_relu, expansion_factor=4):
    with tf.variable_scope(scope):
        inp_channel = x.get_shape()[-1]
        net = slim.conv2d(x, inp_channel * expansion_factor, [1, 1], scope='conv1')
        net = slim.separable_conv2d(net, dim, [ksize, ksize], stride=stride, activation_fn=None, scope='dw_conv2')
        # if stride == 1:
        #     net = net + x
        return net


def ResBottleneckBlock(x, dim, ksize, scope='rb', activation_fn=parametric_relu):
    with tf.variable_scope(scope):
        # net = slim.separable_conv2d(x, dim, [ksize, ksize], scope='dw_conv1')
        # net = slim.conv2d(net, dim, [1, 1], scope='pw_conv1')
        # net = slim.separable_conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='dw_conv2')
        # net = slim.conv2d(net, dim, [1, 1], activation_fn=None, scope='pw_conv2')
        # net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        # net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')

        net = res_bottleneck_dsconv(x, dim, ksize, scope='rb1')
        net = res_bottleneck_dsconv(net, dim, ksize, scope='rb2')
        net = net + x
        return net


def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.separable_conv2d(x, dim, [ksize, ksize], scope='dw_conv1')
        net = slim.separable_conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='dw_conv2')

        net = net + x
        return net