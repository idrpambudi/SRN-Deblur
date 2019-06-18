import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.BasicConvLSTMCell import *

def generator(inputs, reuse=False, scope='g_net', model='lstm', n_levels=3, batch_size=1):
    n, h, w, c = inputs.get_shape().as_list()

    if model == 'lstm':
        with tf.variable_scope('LSTM'):
            cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
            rnn_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    x_unwrap = []
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            inp_pred = inputs
            for i in range(n_levels):
                scale = 0.5 ** (n_levels - i - 1)
                hi = int(round(h * scale))
                wi = int(round(w * scale))
                inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                if model == 'lstm':
                    rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                # encoder
                conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                if model == 'lstm':
                    deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                else:
                    deconv3_4 = conv3_4

                # decoder
                deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                cat2 = deconv2_4 + conv2_4
                deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                cat1 = deconv1_4 + conv1_4
                deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                inp_pred = slim.conv2d(deconv1_1, c, [5, 5], activation_fn=None, scope='dec1_0')

                if i >= 0:
                    x_unwrap.append(inp_pred)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

    return x_unwrap


def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
    return net + x