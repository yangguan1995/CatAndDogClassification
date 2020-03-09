import h5py
import numpy as np
import os
import tensorflow as tf
import keras.backend as k
from keras.layers import BatchNormalization

def Inception_Block(input,c1,c2,c3,c4):
    shape = input.get_shape().as_list()
    with tf.variable_scope('c1-conv'):
        conv1_weights = tf.get_variable('weights',[1,1,shape[3],c1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias',[c1],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input,conv1_weights,strides=[1,1,1,1],padding=None)
        conv1 = BatchNormalization(axis=3)(conv1)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    with tf.variable_scope('c2-conv1'):
        conv1_weights = tf.get_variable('weights',[1,1,shape[3],c2[0]],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias  = tf.get_variable('bias',[c2[0]],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input,conv1_weights,strides=[1,1,1,1],padding=None)
        conv1 = BatchNormalization(axis=3)(conv1)
        relu2_1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))
    with tf.variable_scope('c2-conv2'):
        conv2_weights = tf.get_variable('weights',[3,3,c2[0],c2[1]],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable('bias',[c2[1]],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu2_1,conv2_weights,strides=[1,1,1,1],padding=None)
        conv2 = BatchNormalization(axis=3)(conv2)
        relu2_2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))

    with tf.variable_scope('c3-conv1'):
        conv3_weights = tf.get_variable('weights',[1,1,shape[3],c3[0]],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_bias = tf.get_variable('bias',[c3[1]],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(input,conv3_weights,strides=[1,1,1,1],padding=None)
        conv3 = BatchNormalization(axis=3)(conv3)
        relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_bias))
    with tf.variable_scope('c3-conv2'):
        conv3_weights = tf.get_variable('weights',[5,5,c3[0],c3[1]],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_bias = tf.get_variable('bias',[c3[1]],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(relu3_1,conv3_weights,strides=[1,1,1,1],padding=None)
        conv3 = BatchNormalization(axis=3)(conv3)
        relu3_2 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_bias))

    with tf.variable_scope('c4-maxpool'):
        pool1 = tf.nn.max_pool(input,ksize=[1,3,3,1],strides=[1,1,1,1],padding='same')
    with tf.variable_scope('c4-conv'):
        conv4_weights = tf.get_variable('weights',[1,1,shape[3],c4],initializer=tf.truncated_normal_initializer(stddev=(0.1)))
        conv4_bias = tf.get_variable('bias',[c4],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool1,conv4_weights,strides=[1,1,1,1],padding=None)
        conv4 = BatchNormalization(axis=3)(conv4)
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_bias))

    outputs = tf.concat([relu1,relu2_2,relu3_2,relu4],axis=3)
    return outputs



