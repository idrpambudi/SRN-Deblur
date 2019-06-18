import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.util import *
from util.BasicConvLSTMCell import *

def generator(inputs, scope='g_net',n_levels=2):
    n, h, w, c = inputs.get_shape().as_list()

    x_unwrap = []
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.separable_conv2d],
                            activation_fn=parametric_relu, padding='SAME', normalizer_fn=None,
                        #     activation_fn=parametric_relu, padding='SAME', normalizer_fn=tf.layers.batch_normalization,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # lstm = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', return_sequences=True)
            cell = BasicConvLSTMCell([h / 8, w / 8], [1, 1], 64)
            rnn_state = cell.zero_state(batch_size=n, dtype=tf.float32)
            inp_pred = inputs
            for i in range(n_levels):
                scale = 0.5 ** (n_levels - i - 1)
                hi = int(round(h * scale))
                wi = int(round(w * scale))

                inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                inp_pred = tf.image.resize_images(inp_pred, [hi, wi], method=0)
                inp_pred = tf.stop_gradient(inp_pred)
                inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                rnn_state = tf.image.resize_images(rnn_state, [hi//8, wi//8], method=0)

                # encoder
                # conv1_1 = slim.separable_conv2d(inp_all, 32, [5, 5], scope='enc1_1_dw')

                print(inp_all)
                conv0 = slim.conv2d(inp_all, 8, [5, 5], scope='enc0')
                net = slim.conv2d(conv0, 16, [5, 5], stride=2, scope='enc1_1')
                conv1 = ResBottleneckBlock(net, 16, 5, scope='enc1_2')
                net = res_bottleneck_dsconv(conv1, 32, 5, stride=2, scope='enc2_1')
                net = ResBottleneckBlock(net, 32, 5, scope='enc2_2')
                net = ResBottleneckBlock(net, 32, 5, scope='enc2_3')
                conv2 = ResBottleneckBlock(net, 32, 5, scope='enc2_4')
                net = res_bottleneck_dsconv(conv2, 64, 5, stride=2, scope='enc3_1')
                net = ResBottleneckBlock(net, 64, 5, scope='enc3_2')
                net = ResBottleneckBlock(net, 64, 5, scope='enc3_3')
                net = ResBottleneckBlock(net, 64, 5, scope='enc3_4')
                net = ResBottleneckBlock(net, 64, 5, scope='enc3_5')
                net = ResBottleneckBlock(net, 64, 5, scope='enc3_6')

                net, rnn_state = cell(net, rnn_state)
                # net = lstm(net)
                # decoder
                net = ResBottleneckBlock(net, 64, 5, scope='dec3_6')
                net = ResBottleneckBlock(net, 64, 5, scope='dec3_5')
                net = ResBottleneckBlock(net, 64, 5, scope='dec3_4')
                net = ResBottleneckBlock(net, 64, 5, scope='dec3_3')
                net = ResBottleneckBlock(net, 64, 5, scope='dec3_2')
                net = slim.conv2d_transpose(net, 32, [5, 5], stride=2, scope='dec3_1')
                net = net + conv2
                net = ResBottleneckBlock(net, 32, 5, scope='dec2_4')
                net = ResBottleneckBlock(net, 32, 5, scope='dec2_3')
                net = ResBottleneckBlock(net, 32, 5, scope='dec2_2')
                net = slim.conv2d_transpose(net, 16, [5, 5], stride=2, scope='dec2_1')
                net = net + conv1
                net = ResBottleneckBlock(net, 16, 5, scope='dec1_2')
                net = slim.conv2d_transpose(net, 8, [5, 5], stride=2, scope='dec1_1')
                net = net + conv0
                inp_pred = slim.conv2d(net, c, [5, 5], activation_fn=None, scope='dec0')
                
                x_unwrap.append(inp_pred)
        return x_unwrap