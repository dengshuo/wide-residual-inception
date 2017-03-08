import numpy as np
import tensorflow as tf



DEFAULT_PADDING = 'SAME'


def make_var( name, shape, initializer=None, trainable=True, regularizer=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

def validate_padding(padding):
    assert padding in ('SAME', 'VALID')

def conv( input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
    validate_padding(padding)
    c_i = input.get_shape()[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:

        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                               regularizer=l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
        if biased:
            biases = make_var('biases', [c_o], init_biases, trainable)
            conv = convolve(input, kernel)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)
        else:
            conv = convolve(input, kernel)
            if relu:
                return tf.nn.relu(conv, name=scope.name)
            return conv

def atrous_conv(input, k_h, k_w, c_o,set_rate, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
    validate_padding(padding)
    c_i = input.get_shape()[-1]
    convolve2 = lambda i, k,rate: tf.nn.atrous_conv2d(i, k, rate , padding=padding)
    with tf.variable_scope(name) as scope:

        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                               regularizer=l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
        if biased :
            biases = make_var('biases', [c_o], init_biases, trainable)
            conv = convolve2(input, kernel,set_rate)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)
        else:
            conv = convolve2(input, kernel,set_rate)
            if relu:
                return tf.nn.relu(conv, name=scope.name)
            return conv




def relu( input,name,leakness=0.0):
    if leakness>0.0:
        return tf.maximum(x, x*leakness, name=name)
    else:
        return tf.nn.relu(input, name=name)


def max_pool( input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)



def avg_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)


def fc( input, num_out, name, relu=True, trainable=True):
    with tf.variable_scope(name) as scope:
       
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)

        weights = make_var('weights', [input.get_shape()[1], num_out], init_weights, trainable, \
                                regularizer=l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
        biases = make_var('biases', [num_out], init_biases, trainable)

        fc=tf.nn.bias_add(tf.matmul(input,weights),biases)
        return fcl


def softmax(input, name):

    return tf.nn.softmax(input,name=name)



def add(input,name):
    #pdb.set_trace()
    return tf.add(input[0],input[1])


def batch_normalization(input,name,relu=True,is_training=True):
    if relu:
        temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
        return tf.nn.relu(temp_layer)
    else:
        return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)





def dropout(self, input, keep_prob, name):
    return tf.nn.dropout(input, keep_prob, name=name)

def l2_regularizer(self, weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        #tf.op_scope(values, name, default_name=None)
        with tf.op_scope([tensor], scope, 'l2_regularizer'):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                   dtype=tensor.dtype.base_dtype,
                                   name='weight_decay')
            return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer

 