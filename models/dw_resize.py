import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.util import *

def generator(inputs, scope='g_net',n_levels=3):
    n, h, w, c = inputs.get_shape().as_list()

    x_unwrap = []
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.separable_conv2d],
                            activation_fn=parametric_relu, padding='SAME', normalizer_fn=tf.contrib.layers.instance_norm,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            inp_pred = inputs
            for i in range(n_levels):
                scale = 0.5 ** (n_levels - i - 1)
                hi = int(round(h * scale))
                wi = int(round(w * scale))

                inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                inp_pred = tf.image.resize_images(inp_pred, [hi, wi], method=0)
                inp_pred = tf.stop_gradient(inp_pred)
                inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')

                # encoder
                conv1_1 = slim.separable_conv2d(inp_all, 32, [5, 5], scope='enc1_1_dw')
                conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                conv2_1 = slim.separable_conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1_dw')
                conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                conv3_1 = slim.separable_conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1_dw')
                conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')
                print(conv3_4)

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
                # inp_pred_dw = slim.separable_conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0_dw')
                # inp_pred = slim.conv2d(inp_pred_dw, self.chns, [1, 1], activation_fn=None, scope='dec1_0')
                
                x_unwrap.append(inp_pred)
        return x_unwrap