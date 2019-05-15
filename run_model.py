import os
import argparse
import tensorflow as tf
# import models.model_gray as model
# import models.model_color as model
import models.model as model


def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='test', help='determine whether [test | train | check]')
    parser.add_argument('--datalist', type=str, default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=4)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default='./testing_set',
                        help='input path for testing images')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='output path for testing images')
    parser.add_argument('--step', type=int, default=523000,
                        help='to be retrieved checkpoint step')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='determine continue training from checkpoint')
    parser.add_argument('--dtype', type=str, default='fp32',
                        help='model datatype, using mixed precision if fp16')
                        
    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    os.environ['TF_CPP_VMODULE']="auto_mixed_precision=2"
    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # set up deblur models
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.test(args.height, args.width, args.input_path, args.output_path, args.step)
    elif args.phase == 'train':
        if args.continue_training:
            deblur.train(checkpoint_step=args.step)
        else:
            deblur.train()
    elif args.phase == 'check':
        deblur.check(args.height, args.width)
    elif args.phase == 'convert':
        deblur.convert_tflite(args.step, args.height, args.width)
    elif args.phase == 'eval':
        deblur.convert_tflite(args.step)
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()