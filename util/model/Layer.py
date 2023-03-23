import tensorflow as tf
import numpy as np


def conv_layer(input, input_channel, output_channel, mean=0.0, std=1., bias=0.0, filter_size=3, name=None, trainable=True, padding="SAME", strides=1):

    if name is None:
        name = 'conv_layer'
        
    with tf.variable_scope(name):
        
        shape = [filter_size, filter_size, input_channel, output_channel]
    
        with tf.device('/cpu:0'):
            W = tf.get_variable(name="W", trainable=trainable, initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=std))
            B = tf.get_variable(name="B", trainable=trainable, initializer=tf.constant(bias, shape=[output_channel]))
    
        conv = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding=padding)
        conv = tf.nn.bias_add(conv, B)

    return conv


def deconv_layer(input, input_channel, output_channel, mean=0.0, std=1., bias=0.0, filter_size=3, stride=2, name=None, batch_size=1):

    if name is None:
        name = 'deconv_layer'
        
    with tf.variable_scope(name):
        output_shape = [batch_size, int(input.shape[1] * stride), int(input.shape[2] * stride), output_channel]
        shape = [filter_size, filter_size, output_channel, input_channel]
    
        
        with tf.device('/cpu:0'):
            W = tf.get_variable(name='W', trainable=True, initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=std), dtype=tf.float32)
            B = tf.get_variable(name='B', trainable=True, initializer=tf.constant(bias, shape=[output_channel]), dtype=tf.float32)
    
        conv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, B)
    
        return conv


def max_pooling(input, size=2, name=None):
    if name is None:
        name='maxpooling'
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding="SAME", name=name)


def avg_pooling(input, size=2):
    return tf.nn.avg_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding="SAME")


def batch_norm(x, n_out, decay=0.99, eps=1e-5, name=None, trainable=True): # n_out=필터의 수
    
    if name is None:
        name = 'norm'
        
    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                                   , trainable=trainable)
            gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                    trainable=trainable)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments') # 평균과 분산 계산
        ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(tf.constant(True), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    return normed


def group_norm(x, batch, G, gamma=None, beta=None, eps=1e-5, name='grouppNorm', trainable=True):

    _, H, W, C = x.shape

    if gamma is None:
        gamma = tf.get_variable(name="%s_gamma" % name, trainable=trainable, initializer=np.ones(shape=[1, 1, 1, C], dtype=np.float32))

    if beta is None:
        beta = tf.get_variable(name="%s_beta" % name, trainable=trainable, initializer=np.zeros(shape=[1, 1, 1, C], dtype=np.float32))

    x = tf.reshape(x, [batch, H, W, C // G, G])

    mean, var = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [batch, H, W, C])

    return x * gamma + beta

