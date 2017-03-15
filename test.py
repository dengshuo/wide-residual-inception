#!/usr/bin/env python

import os
from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import cifar10_input as data_input
import wrinc
import utils
import pdb



# Dataset Configuration
tf.app.flags.DEFINE_string('data_dir', './cifar-10-batches-bin', """Path to the CIFAR-100 binary data.""")
tf.app.flags.DEFINE_integer('num_classes', 10, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 50000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")


tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 100.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

# tf.app.flags.DEFINE_boolean('finetune', False,
                            # """Flag whether the L1 connection weights will be only made at
                            # the position where the original bypass network has nonzero
                            # L1 connection weights""")
# tf.app.flags.DEFINE_string('pretrained_dir', './pretrain', """Directory where to load pretrained model.(Only for --finetune True""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 360000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 3000, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 5000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tCIFAR-10 dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    #print('\tResidual blocks per group: %d' % FLAGS.num_residual_units)
    #print('\tNetwork width multiplier: %d' % FLAGS.k)

    print('[Optimization Configuration]')
    #print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %f' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per testing: %d' % FLAGS.test_interval)
    print('\tSteps during testing: %d' % FLAGS.test_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of CIFAR-100
        with tf.variable_scope('train_image'):
            train_images, train_labels = data_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.inputs(True, FLAGS.data_dir, FLAGS.batch_size)

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.IMAGE_SIZE, data_input.IMAGE_SIZE, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Build model
        decay_step = FLAGS.lr_step_epoch * FLAGS.num_train_instance / FLAGS.batch_size
        hp = wrinc.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            initial_lr=FLAGS.initial_lr,
                            decay_step=decay_step,
                            lr_decay=FLAGS.lr_decay,
                            momentum=FLAGS.momentum)
        network = wrinc.wrinc(hp, images, labels, global_step)
        network.build_model()
        network.build_train_op()

        # Summaries(training)
        train_summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        #pdb.set_trace()
        if ckpt and ckpt.model_checkpoint_path:
           print('\tRestore from %s' % ckpt.model_checkpoint_path)
           # Restores from checkpoint
           saver.restore(sess, ckpt.model_checkpoint_path)
           init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
           print('No checkpoint file found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        # Training!
        test_loss, test_acc = 0.0, 0.0
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images, test_labels])
            loss_value, acc_value = sess.run([network.loss,network.acc],
                        feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
            test_loss += loss_value
            test_acc += acc_value
        test_loss /= FLAGS.test_iter
        test_acc /= FLAGS.test_iter
        test_best_acc = max(test_best_acc, test_acc)
        format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f')
        print (format_str % (datetime.now(), step, test_loss, test_acc))
