import numpy as np
import tensorflow as tf
import pdb

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv

def _fc(x, out_dim, name='fc'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/out_dim)))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc


# def _bn(x, is_train, global_step=None, name='bn'):
#     moving_average_decay = 0.9
#     # moving_average_decay = 0.99
#     # moving_average_decay_init = 0.99
#     with tf.variable_scope(name):
#         decay = moving_average_decay
#         # if global_step is None:
#             # decay = moving_average_decay
#         # else:
#             # decay = tf.cond(tf.greater(global_step, 100)
#                             # , lambda: tf.constant(moving_average_decay, tf.float32)
#                             # , lambda: tf.constant(moving_average_decay_init, tf.float32))
#         batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
#         mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
#                         initializer=tf.zeros_initializer, trainable=False)
#         sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
#                         initializer=tf.ones_initializer, trainable=False)
#         beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
#                         initializer=tf.zeros_initializer)
#         gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
#                         initializer=tf.ones_initializer)
#         # BN when training
#         update = 1.0 - decay
#         # with tf.control_dependencies([tf.Print(decay, [decay])]):
#             # update_mu = mu.assign_sub(update*(mu - batch_mean))
#         update_mu = mu.assign_sub(update*(mu - batch_mean))
#         update_sigma = sigma.assign_sub(update*(sigma - batch_var))
#         tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
#         tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

#         mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
#                             lambda: (mu, sigma))
#         bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

#         # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

#         # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
#                                           # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
#                                           # scale=True, epsilon=1e-5, is_training=is_train,
#                                           # trainable=True)
#     return bn

def _bn(input,is_train,global_step=None,name='bn'):
        return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_train,scope=name)

## Other helper functions


def _avg_pool(input,padding,name):
	filter_size = input.get_shape()[1]
	return tf.nn.avg_pool(input,[1,filter_size,filter_size,1],[1,1,1,1],padding=padding,name=name)



def _residual_mod(input,filter_size,kernel_num,has_side_conv,is_stride,is_train,global_step,name):

    if is_stride:
        conv_add=_conv(input,1,kernel_num,2,name=(name+'_0_conv'))
        conv_add_bn=_bn(conv_add,is_train,global_step,name=(name+'_0_bn'))
        first_stride=2
    else:
        first_stride=1
        if has_side_conv:
            conv_add=_conv(input,1,kernel_num,1,name=(name+'_0_conv'))
            conv_add_bn=_bn(conv_add,is_train,global_step,name=(name+'_0_bn'))
        else :
            conv_add_bn=input

    conv_out=_conv(input,filter_size,kernel_num,first_stride,name=(name+'_1_conv'))
    conv_bn=_bn(conv_out,is_train,global_step,name=((name)+'_1_bn'))
    conv_relu=_relu(conv_bn,name=((name)+'_1_relu'))

    conv_out_2=_conv(conv_relu,filter_size,kernel_num,1,name=(name+'_2_conv'))
    conv_bn_2=_bn(conv_out_2,is_train,global_step,name=((name)+'_2_bn'))
    #conv_relu_2=_relu(conv_bn_2,name=((basename)+'_2_relu'))



    out=tf.add(conv_add_bn,conv_bn_2,name=((name)+'_residual_add'))
    out=_relu(out,name=((name)+'_residual_bn'))
    return out


def _inception1(input,filter_size,kernel_num,is_train,global_step,name):
    conv_1=_conv(input,filter_size[0],kernel_num[0],1,name=(name+'input_conv'))
    conv_1_bn=_bn(conv_1,is_train,global_step,name=(name+'input_bn'))
    conv_1_relu=_relu(conv_1_bn,name=(name+'input_relu'))



    conv_a_1=_conv(conv_1_relu,filter_size[1],kernel_num[1],1,name=(name+'conv_a_1'))
    conv_a_1_bn=_bn(conv_a_1,is_train,global_step,name=(name+'conv_a_1_bn'))
    conv_a_1_relu=_relu(conv_a_1_bn,name=(name+'conv_a_relu'))

    conv_a_2=_conv(conv_a_1_relu,filter_size[2],kernel_num[2],1,name=(name+'conv_a_2'))
    conv_a_2_bn=_bn(conv_a_2,is_train,global_step,name=(name+'conv_a_2_bn'))
    conv_a=_relu(conv_a_2_bn,name=(name+'conv_a'))


    conv_b_conv=_conv(conv_1_relu,filter_size[3],kernel_num[3],1,name=(name+'conv_b_conv'))
    conv_b_bn=_bn(conv_b_conv,is_train,global_step,name=(name+'conv_b_2_bn'))
    conv_b=_relu(conv_b_bn,name=(name+'conv_b'))




    print conv_1_relu.get_shape(), conv_a.get_shape(), conv_b.get_shape()
    out=tf.concat(3,[conv_1_relu,conv_a,conv_b])
    out=_conv(out,filter_size[4],kernel_num[4],1,name=(name+'out_conv'))
    out=_bn(out,is_train,global_step,name=(name+'out_bn'))
    out=_relu(out,name=(name+'out_relu'))

    print out.get_shape()

    return out





