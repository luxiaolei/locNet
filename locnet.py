

import numpy as np 
import tensorflow as tf


class LocNet: 
    def __init__(self, scope, buttom_layer):
        self.scope = scope 
        with tf.variable_scope(scope) as scope:
            self.build_graph(buttom_layer)
            self.gt_loc = tf.placeholder(dtype=tf.float32, shape=(None,4),name='gt_loc')
        
    def build_graph(self, buttom_layer):
        self.variables = []
        self.kernel_weights = []
        pool = tf.nn.max_pool(buttom_layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool')
        
        drop = tf.nn.dropout(pool, 0.3)
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(drop.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 3000],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[3000], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool_flat = tf.reshape(drop, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)
            self.kernel_weights += [fc1w]
            self.variables += [fc1w, fc1b]
            

        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([3000, 4],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.logit = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            self.kernel_weights += [fc2w]
            self.variables += [fc2w, fc2b]
            
    def loss(self):
        with tf.name_scope(self.scope) as scope:
            beta = tf.constant(0.05, name='beta')
            loss_rms = tf.reduce_max(tf.squared_difference(self.gt_loc, self.logit))
            loss_wd = [tf.reduce_mean(tf.square(w)) for w in self.kernel_weights]
            loss_wd = beta * tf.add_n(loss_wd)
            total_loss = loss_rms + loss_wd
        return total_loss
