from __future__ import print_function
import os
import time
import random
import datetime
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm
from datetime import datetime
from util.util import *
from util.var_storage import fp32_trainable_vars
from util.BasicConvLSTMCell import *


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.dtype = tf.float32 if args.dtype == 'fp32' else tf.float16
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        print(len(self.data_list))
        print(self.data_size)

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                                  axis=0)
            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=self.dtype)

        x_unwrap = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.separable_conv2d],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in range(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))

                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_blur = tf.cast(inp_blur, self.dtype)
                    print(inp_pred.dtype)
                    inp_pred = tf.image.resize_images(inp_pred, [hi, wi], method=0)
                    inp_pred = tf.cast(inp_pred, self.dtype)
                    print(inp_pred.dtype)
                    inp_pred = tf.stop_gradient(inp_pred)
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')

                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

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

                    if self.args.model == 'lstm':
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
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')
                    # inp_pred_dw = slim.separable_conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0_dw')
                    # inp_pred = slim.conv2d(inp_pred_dw, self.chns, [1, 1], activation_fn=None, scope='dec1_0')
                    
                    x_unwrap.append(inp_pred)
            return x_unwrap

    def build_model(self):
        img_in, img_gt = self.input_producer(self.batch_size)
        img_in = tf.cast(img_in, self.dtype)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        # generator
        # with fp32_trainable_vars():
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        # x_unwrap = [tf.cast(x, tf.float32) for x in x_unwrap]

        # calculate multi-scale loss
        self.loss_total = 0
        for i in range(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self, checkpoint_step=0):
        def get_optimizer(loss, var_list=None, global_step=tf.Variable(initial_value=0, dtype=tf.int32, trainable=False), is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)

            if self.dtype == tf.float16:
                loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(2**12)
                train_op = tf.contrib.mixed_precision.LossScaleOptimizer(train_op, loss_scale_manager)

            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = checkpoint_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(
            self.learning_rate, global_step, self.max_steps, end_learning_rate=self.learning_rate*1e-2, power=0.9)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        
        if checkpoint_step > 0:
            self.load(sess, self.train_dir, step=checkpoint_step)
        else:
            sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for ep in range(1, self.epoch + 1):
            epoch_loss = 0
            epoch_start_time = time.time()

            iterator = tqdm(range(self.data_size), leave=False, desc='Epoch {}'.format(ep), ncols=100)
            for step in iterator:
                # update G network
                _, loss_total_val = sess.run([train_gnet, self.loss_total])
                assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'
                epoch_loss += loss_total_val
                global_step += 1

                iterator.set_postfix({'step_loss': loss_total_val})

            epoch_duration = time.time() - epoch_start_time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            mean_epoch_loss = epoch_loss * self.batch_size / len(self.data_list)
            data_per_second = len(self.data_list) / epoch_duration

            epoch_str = 'Epoch {}: average_loss = {:.5f} ({:.1f} data/s; {:.2f} s/epoch), {}'.format(
                ep, mean_epoch_loss, data_per_second, epoch_duration, current_time)
            tqdm.write(epoch_str)
   
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=global_step)
            
            # Save the model checkpoint periodically every 5 epoch.
            if ep % 5 == 0 or ep == self.epoch:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, ep)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def eval(self, step, height=720, width=1280, file_dir='eval_set'):
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, height, width, inp_chns], dtype=self.dtype)
        img_gt = tf.placeholder(shape=[self.batch_size, height, width, inp_chns], dtype=self.dtype)
        outputs = self.generator(inputs, reuse=False)
    
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=step)

        # calculate multi-scale loss
        loss_total = 0
        for i in range(self.n_levels):
            _, hi, wi, _ = outputs[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - outputs[i]) ** 2)
            loss_total += loss
        
        eval_loss = 0
        iterator = tqdm(range(len(self.data_list)))
        for i in iterator:
            sharp = cv2.imread(os.path.join(file_dir, self.data_list[i][0]))
            sharp = np.expand_dims(sharp, axis=0)
            blur = cv2.imread(os.path.join(file_dir, self.data_list[i][1]))
            blur = np.expand_dims(blur, axis=0)
            loss = sess.run(loss_total, feed_dict={inputs: blur/255.0, img_gt: sharp/255.0})
            eval_loss += loss
            iterator.set_postfix_str('loss = {}'.format(loss))
        print(eval_loss / len(self.data_list))

    def test(self, height, width, input_path, output_path, step):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=self.dtype)
        outputs = self.generator(inputs, reuse=False)
    
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=step)

        for imgName in imgsName:
            blur = cv2.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = cv2.resize(blur, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            cv2.imwrite(os.path.join(output_path, imgName), res)


    def check(self, height, width):
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        run_meta = tf.RunMetadata()
        with sess:
            H, W = height, width
            inp_chns = 3 if self.args.model == 'color' else 1
            self.batch_size = 1 if self.args.model == 'color' else 3
            inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=self.dtype)
            outputs = self.generator(inputs, reuse=False)

            opts = tf.profiler.ProfileOptionBuilder.float_operation()    
            flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
            params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
            print('Total FLOPs, Total Parameters')
            print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))


    def convert_tflite(self, step, height, width):
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, height, width, inp_chns], dtype=self.dtype)
        outputs = self.generator(inputs, reuse=False)

        print(inputs.name)
        print(outputs[-1].name)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=step)
        # converter = tf.lite.TFLiteConverter.from_session(sess, [inputs], [outputs[-1]])

        saved_model_dir = './checkpoints/saved_model'
        builder = tf.saved_model.Builder(saved_model_dir)
        tensor_info_input = tf.saved_model.utils.build_tensor_info(inputs)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(outputs[-1])
        prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'images': tensor_info_input},
            outputs={'scores': tensor_info_output}
        )
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'serving_default': prediction_signature}
        )
        builder.save(as_text=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        tflite_model = converter.convert()
        open("converted_model-{}.tflite".format(step), "wb").write(tflite_model)