from __future__ import print_function
import os
import shutil
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
from models.dw_smaller import generator


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 2
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.dtype = tf.float32
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.checkpoint = args.checkpoint
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.generator = generator

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


    def build_trainer(self):
        img_in, img_gt = self.input_producer(self.batch_size)
        img_in = tf.cast(img_in, self.dtype)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        x_unwrap = self.generator(img_in, scope='g_net')

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

    def train(self):
        def get_optimizer(loss, var_list, global_step=tf.Variable(initial_value=self.checkpoint, dtype=tf.int32, trainable=False)):
            train_op = tf.train.AdamOptimizer(self.lr)
            train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = self.checkpoint * self.data_size

        # build model
        self.build_trainer()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(
            self.learning_rate, global_step, self.max_steps, end_learning_rate=1e-6, power=0.4)

        # g = tf.get_default_graph()
        # tf.contrib.quantize.create_training_graph(input_graph=g,quant_delay=200000)

        # training operators
        train_gnet = get_optimizer(self.loss_total, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        
        if self.checkpoint > 0:
            self.load(sess, self.train_dir, epoch=self.checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for ep in range(self.checkpoint + 1, self.epoch + 1):
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
            average_loss = epoch_loss * self.batch_size / len(self.data_list)
            data_per_second = len(self.data_list) / epoch_duration

            epoch_str = 'Epoch {}: average_loss = {:.5f} ({:.1f} data/s; {:.2f} s/epoch), {}'.format(
                ep, average_loss, data_per_second, epoch_duration, current_time)
            tqdm.write(epoch_str)
            with open('checkpoints/log.txt','a+') as f:
                print(epoch_str, file=f, flush=True)
   
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=global_step)
            
            # Save the model checkpoint periodically every 5 epoch.
            if ep % 5 == 0 or ep == self.epoch:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, ep)

    def save(self, sess, checkpoint_dir, epoch):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=epoch)

    def load(self, sess, checkpoint_dir, epoch=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if epoch is not None:
            ckpt_name = model_name + '-' + str(epoch)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(epoch)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def eval(self, height=720, width=1280, file_dir='training_set'):
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, height, width, inp_chns], dtype=self.dtype)
        img_gt = tf.placeholder(shape=[self.batch_size, height, width, inp_chns], dtype=self.dtype)
        outputs = self.generator(inputs)
    
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, epoch=self.checkpoint)

        # calculate multi-scale loss
        loss_total = 0
        for i in range(self.n_levels):
            _, hi, wi, _ = outputs[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - outputs[i]) ** 2)
            loss_total += loss
            
        psnr_tensor = tf.image.psnr(outputs[-1], img_gt, max_val=1.0)
        ssim_tensor = tf.image.ssim(outputs[-1], img_gt, max_val=1.0)
        
        avg_psnr = 0
        avg_ssim = 0
        eval_loss = 0
        iterator = tqdm(range(len(self.data_list)))
        for i in iterator:
            sharp = cv2.imread(os.path.join(file_dir, self.data_list[i][0]))
            sharp = np.expand_dims(sharp, axis=0)
            blur = cv2.imread(os.path.join(file_dir, self.data_list[i][1]))
            blur = np.expand_dims(blur, axis=0)
            deblur_result, loss, psnr, ssim = sess.run([outputs, loss_total, psnr_tensor, ssim_tensor], feed_dict={inputs: blur/255.0, img_gt: sharp/255.0})
            
            eval_loss += loss
            avg_psnr += psnr
            avg_ssim += ssim
            iterator.set_postfix({'loss': loss, 'ssim':ssim, 'psnr':psnr})

        eval_loss /= len(self.data_list)
        avg_psnr /= len(self.data_list)
        avg_ssim /= len(self.data_list)
        print({'loss':eval_loss, 'ssim':avg_ssim, 'psnr':avg_psnr})

    def test(self, height, width, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=self.dtype)
        outputs = self.generator(inputs)
    
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())

        # self.saver = tf.train.Saver()
        # self.load(sess, self.train_dir, epoch=self.checkpoint)

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
            outputs = self.generator(inputs)

            opts = tf.profiler.ProfileOptionBuilder.float_operation()    
            flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
            params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
            print('Total FLOPs, Total Parameters')
            print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))


    def convert_tflite(self, height, width, is_quantize=True):
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, height, width, inp_chns], dtype=self.dtype)
        outputs = self.generator(inputs)

        print(inputs.name)
        print(outputs[-1].name)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)

        self.saver = tf.train.Saver()
        # self.load(sess, self.train_dir, epoch=self.checkpoint)
        sess.run(tf.global_variables_initializer())

        # converter = tf.lite.TFLiteConverter.from_session(sess, [inputs], [outputs[-1]])

        saved_model_dir = './checkpoints/saved_model'
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
        builder = tf.saved_model.Builder(saved_model_dir)
        tensor_info_input = tf.saved_model.utils.build_tensor_info(inputs)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(outputs[-1])
        prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'images': inputs},
            outputs={'scores': outputs[-1]}
        )
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'serving_default': prediction_signature}
        )
        builder.save(as_text=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        converter.post_training_quantize = is_quantize
        tflite_model = converter.convert()

        open("converted_model-{}.tflite".format(step), "wb").write(tflite_model)

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()