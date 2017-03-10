from collections import namedtuple

import tensorflow as tf
import numpy as np
import pdb

import utils


HParams = namedtuple('HParams',
                    'batch_size, num_classes, '
                    'initial_lr, decay_step, lr_decay, '
                    'momentum')






class wrinc(object):
    def __init__(self, hp, images, labels, global_step):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels
        self._global_step = global_step
        self.is_train = tf.placeholder(tf.bool)

    def build_model(self):
        print('Building model')
        # Init. conv.
        print('\tBuilding unit: conv1')
        conv_1 = utils._conv(self._images, 3, 64, 1, name='conv_1')
        conv_1_bn = utils._bn(conv_1, self.is_train, self._global_step, name='conv_1_bn')
        conv1_relu = utils._relu(conv_1_bn, name='conv1_relu')

        # Residual Blocks
        #_residual_mod(input,filter_size,kernel_num,has_side_conv,is_stride,is_train,global_step,name=basename)

        with tf.variable_scope('conv2' ) as scope:

            conv2_1=utils._residual_mod(conv1_relu,3,64,True,False,self.is_train,self._global_step,name='conv2_1')
            conv2_2=utils._residual_mod(conv2_1,3,64,False,False,self.is_train,self._global_step,name='conv2_2')

        with tf.variable_scope('conv3' ) as scope:
            conv3=utils._residual_mod(conv2_2,3,256,False,True,self.is_train,self._global_step,name='conv3')

#_inception1(input,filter_size,kernel_num,is_train,global_step,name=basename):
        with tf.variable_scope('inception' ) as scope:
            inception=utils._inception1(conv3,[1,3,3,3,1],[256,128,256,128,256],self.is_train,self._global_step,name='inception')


        with tf.variable_scope('conv4' ) as scope:
            conv4=utils._residual_mod(inception,3,256,False,True,self.is_train,self._global_step,name='conv4')
            conv4_ave_pool=utils._avg_pool(conv4,'VALID',name='conv4_ave_pool')
        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            conv4_ave_pool_shape = conv4_ave_pool.get_shape().as_list()
            dim_conv4_ave_pool=conv4_ave_pool_shape[1]*conv4_ave_pool_shape[2]*conv4_ave_pool_shape[3]
            x = tf.reshape(conv4_ave_pool, [conv4_ave_pool_shape[0],dim_conv4_ave_pool ])
            #pdb.set_trace()
            x = utils._fc(x, self._hp.num_classes)

        print x.get_shape()
        #pdb.set_trace()

        self._logits = x


        self.probs = tf.nn.softmax(x, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.select(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        #tf.scalar_summary('accuracy', self.acc)


        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(x, self._labels)
        self.loss = tf.reduce_mean(loss, name='cross_entropy')
        #tf.scalar_summary('cross_entropy', self.loss)


    def build_train_op(self):

        self._total_loss = self.loss 

        # Learning rate
        self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        self._hp.decay_step, self._hp.lr_decay, staircase=True)
        #tf.scalar_summary('learing_rate', self.lr)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        # print '\n'.join([t.name for t in tf.trainable_variables()])
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops+[apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op
