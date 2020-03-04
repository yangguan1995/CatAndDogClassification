import h5py
import numpy as np
import os
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, \
    Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda,MaxPooling2D

# Load hdf5 dataset

def load_data(data_dir,dataname):
    train_dataset = h5py.File(os.path.join(data_dir,dataname), 'r')
    train_set_x_orig = np.array(train_dataset['X_train'][:]) # your train set features
    train_set_x_orig = train_set_x_orig/255.
    train_set_y_orig = np.array(train_dataset['y_train'][:]) # your train set labels
    test_set_x_orig = np.array(train_dataset['X_test'][:]) # your train set features
    test_set_x_orig = test_set_x_orig/255.
    test_set_y_orig = np.array(train_dataset['y_test'][:]) # your train set labels f.close()
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig

def one_hot_matrix(labels, C):
    C = tf.constant(C,name = "C")
    one_hot_matrix = tf.one_hot(labels,C,axis = -1)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

def conv_add_pool_2d(input_tensor,shape,scope_name):
    with tf.variable_scope(name_or_scope=scope_name):
        conv_weight = tf.get_variable('weight',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',shape=[shape[-1]],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input=input_tensor,filter=conv_weight, strides=[1,1,1,1],padding='SAME')
        #conv = BatchNormalization(axis=3)(conv)
        relu = tf.nn.relu(tf.nn.bias_add(conv,bias))
        pool = tf.nn.max_pool(relu,[1,2,2,1],[1,2,2,1],padding='VALID')
    return pool

##add SEBlock CNN
def conv_add_SE_pool(input_tensor,shape,scope_name):
    with tf.variable_scope(name_or_scope=scope_name):
        conv_weight = tf.get_variable('weight',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',shape=[shape[-1]],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input=input_tensor,filter=conv_weight, strides=[1,1,1,1],padding='SAME')
        conv = BatchNormalization(axis=3)(conv)
        relu = tf.nn.relu(tf.nn.bias_add(conv,bias))
        SE = SEblock(relu,16)
        pool = tf.nn.max_pool(SE,[1,2,2,1],[1,2,2,1],padding='VALID')
    return pool

def fc(input_tensor,shape,scope_name):
    with tf.variable_scope(name_or_scope=scope_name):
        fc_weight = tf.get_variable('weight',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',shape=[shape[-1]],initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input_tensor,fc_weight)+bias
        fc = BatchNormalization(axis=1)(fc)
        relu = tf.nn.relu(fc)
    return  relu

def last_layer(input_tensor,shape,scope_name):
    with tf.variable_scope(name_or_scope=scope_name):
        weight = tf.get_variable('weight',shape = shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',shape = [shape[-1]],initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input_tensor,weight)+bias
        logit = BatchNormalization(axis=1)(fc)
    return logit

def SEblock(input,scale):
    shape = input.get_shape().as_list()
    poolsize = [1,shape[1],shape[2],1]
    GP = tf.nn.avg_pool(value=input,ksize=poolsize,strides=[1,1,1,1],padding='VALID')
    pool_shape = GP.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    channel = pool_shape[3]
    GP = tf.layers.flatten(GP)
    FC1 = fc(GP,[nodes,channel/scale],scope_name='SE_FC1')
    with tf.variable_scope('fc2-sigmoid'):
        fc2_weight = tf.get_variable('weight',shape=[channel/scale,channel],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('bias',shape=[channel],initializer=tf.constant_initializer(0.0))
        fc2 = tf.matmul(FC1,fc2_weight)+fc2_bias
        logit = tf.nn.sigmoid(fc2)
        mask = tf.reshape(logit,[-1,1,1,channel])
    return input*mask

def CBAM_Channel(input,scale):
    shape = input.get_shape().as_list()
    poolsize = [1,shape[1],shape[2],1]
    ##avgpooling channel
    avg = tf.nn.avg_pool(value=input,ksize=poolsize,strides=[1,1,1,1],padding='VALID')
    avg_shape = avg.get_shape().as_list()
    nodes_avg = avg_shape[1]*avg_shape[2]*avg_shape[3]
    channel_avg = avg_shape[3]
    avg = tf.layers.flatten(avg)
    FC_avg = fc(avg,[nodes_avg,channel_avg/scale],scope_name='avg_FC1')
    with tf.variable_scope('fc2-avg'):
        fc2_weight = tf.get_variable('weight',shape=[channel_avg/scale,channel_avg],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('bias',shape=[channel_avg],initializer=tf.constant_initializer(0.0))
        fc2_avg = tf.matmul(FC_avg,fc2_weight)+fc2_bias
    ##maxpooling channel
    max = tf.nn.max_pool(value=input, ksize=poolsize, strides=[1,1,1,1], padding='VALID')
    max_shape = max.get_shape().as_list()
    nodes_max = max_shape[1]*max_shape[2]*max_shape[3]
    channel_max = max_shape[3]
    max = tf.layers.flatten(max)
    FC_max = fc(avg,[nodes_max,channel_max/scale],scope_name='max_FC1')
    with tf.variable_scope('fc2-max'):
        fc2_weight = tf.get_variable('weight',shape=[channel_max/scale,channel_max],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('bias',shape=[channel_max],initializer=tf.constant_initializer(0.0))
        fc2_max = tf.matmul(FC_max,fc2_weight)+fc2_bias
    # add avg and max channel
    logit = tf.nn.sigmoid(fc2_avg+fc2_max)
    mask = tf.reshape(logit, [-1, 1, 1, channel_max])
    return tf.multiply(input,mask)

def CBAM_Spatial(input):
    shape = input.get_shape().as_list()
    avg = tf.reduce_mean(input_tensor=input,axis=-1,keep_dims=True)
    max = tf.reduce_max(input_tensor=input,axis=-1,keep_dims=True)
    add_max_avg = tf.concat([avg,max],3)
    add_max_avg = tf.reshape(add_max_avg,[-1,shape[1],shape[2],2])
    assert add_max_avg.get_shape().as_list()==[None,shape[1],shape[2],2]
    with tf.variable_scope('conv_77'):
        weight = tf.get_variable('weight',shape=[7,7,2,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(add_max_avg,weight,strides=[1,1,1,1],padding='SAME')
    sig = tf.sigmoid(conv)
    mask = tf.reshape(sig, [-1, shape[1], shape[2], 1])
    return tf.multiply(input,mask)

def conv_add_CBAM_pool(input_tensor,shape,scope_name):
    with tf.variable_scope(name_or_scope=scope_name):
        conv_weight = tf.get_variable('weight',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',shape=[shape[-1]],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input=input_tensor,filter=conv_weight, strides=[1,1,1,1],padding='SAME')
        #conv = BatchNormalization(axis=3)(conv)
        relu = tf.nn.relu(tf.nn.bias_add(conv,bias))
        Channel = CBAM_Channel(relu,16)
        Spatial = CBAM_Spatial(Channel)
        pool = tf.nn.max_pool(Spatial,[1,2,2,1],[1,2,2,1],padding='VALID')
    return pool

####写一个Inceptioon的函数
def InceptionBlock(input_tensor,c1,c2,c3,c4):
    x1 = Conv2D(filters=c1,kernel_size=(1,1))(input_tensor)
    x2_1 = Conv2D(filters=c2[0],kernel_size=(1,1))(input_tensor)
    x2_2 = Conv2D(filters=c2[1],kernel_size=(3,3),padding='same')(x2_1)
    x3_1 = Conv2D(filters=c3[0],kernel_size=(1,1))(input_tensor)
    x3_2 = Conv2D(filters=c3[1],kernel_size=(5,5),padding='same')(x3_1)
    x4_1 = MaxPooling2D(pool_size=(3,3),padding='same')(input_tensor)
    x4_2 = Conv2D(filters=c4,kernel_size=(1,1))(x4_1)
    
    return  x4_2











