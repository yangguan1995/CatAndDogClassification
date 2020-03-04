import tensorflow as tf
import numpy as np
from keras.layers import BatchNormalization
from utils import *
import os

tf.reset_default_graph()

def cat_dog_model(input_data):
    conv1 = conv_add_pool_2d(input_data,[3,3,3,32],scope_name='conv1')
    conv2 = conv_add_pool_2d(conv1,[3,3,32,32],scope_name='conv2')
    conv3 = conv_add_pool_2d(conv2,[3,3,32,64],scope_name='conv3')
    conv4 = conv_add_pool_2d(conv3,[3,3,64,128],scope_name='conv4')
    pool_shape = conv4.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    flatten = tf.layers.flatten(conv4)
    fc1 = fc(flatten,[nodes,128],scope_name='fc1')
    logit = last_layer(fc1, [128,2], scope_name='last_layer')
    return logit
def cat_dog_SEmodel(input_data):
    conv1 = conv_add_SE_pool(input_data,[3,3,3,32],scope_name='conv1')
    conv2 = conv_add_SE_pool(conv1,[3,3,32,32],scope_name='conv2')
    conv3 = conv_add_SE_pool(conv2,[3,3,32,64],scope_name='conv3')
    conv4 = conv_add_SE_pool(conv3,[3,3,64,128],scope_name='conv4')
    pool_shape = conv4.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    flatten = tf.layers.flatten(conv4)
    fc1 = fc(flatten,[nodes,128],scope_name='fc1')
    logit = last_layer(fc1, [128,2], scope_name='last_layer')
    return logit
def cat_dog_CBAMmodel(input_data):
    conv1 = conv_add_CBAM_pool(input_data,[3,3,3,32],scope_name='conv1')
    conv2 = conv_add_CBAM_pool(conv1,[3,3,32,32],scope_name='conv2')
    conv3 = conv_add_CBAM_pool(conv2,[3,3,32,64],scope_name='conv3')
    conv4 = conv_add_CBAM_pool(conv3,[3,3,64,128],scope_name='conv4')
    pool_shape = conv4.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    flatten = tf.layers.flatten(conv4)
    fc1 = fc(flatten,[nodes,128],scope_name='fc_1')
    logit = last_layer(fc1, [128,2], scope_name='last_layer')
    return logit
# def cat_dog_model(input,train,regularizer):
#     with tf.variable_scope('conv1'):
#         conv1_weight = tf.get_variable('weight',[3,3,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
#         conv1_biase = tf.get_variable('biases',[32],initializer=tf.constant_initializer(0.))
#         conv1 = tf.nn.conv2d(input,conv1_weight,strides = [1,1,1,1],padding = 'SAME')
#         conv1 = BatchNormalization(axis=3)(conv1)
#         relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biase))
#     with tf.variable_scope('maxpool1'):
#         pool1 = tf.nn.max_pool(relu1,[1,2,2,1],[1,2,2,1],padding='VALID')
#     with tf.variable_scope('conv2'):
#         conv2_weight = tf.get_variable('weight',[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
#         conv2_biase = tf.get_variable('biases',[32],initializer=tf.constant_initializer(0.))
#         conv2 = tf.nn.conv2d(pool1,conv2_weight,strides = [1,1,1,1],padding = 'SAME')
#         conv2 = BatchNormalization(axis=3)(conv2)
#         relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biase))
#     with tf.variable_scope('maxpool2'):
#         pool2 = tf.nn.max_pool(relu2,[1,2,2,1],[1,2,2,1],padding='VALID')
#     with tf.variable_scope('conv3'):
#         conv3_weight = tf.get_variable('weight',[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
#         conv3_bias = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.))
#         conv3 = tf.nn.conv2d(pool2,conv3_weight,strides=[1,1,1,1],padding='SAME')
#         conv3 = BatchNormalization(axis=3)(conv3)
#         relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_bias))
#     with tf.variable_scope('maxpool3'):
#         pool3 = tf.nn.max_pool(relu3,[1,2,2,1],[1,2,2,1],padding='VALID')
#     with tf.variable_scope('conv4'):
#         conv4_weight = tf.get_variable('weight',[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
#         conv4_bias = tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.0))
#         conv4 = tf.nn.conv2d(pool3,conv4_weight,strides=[1,1,1,1],padding='SAME')
#         conv4 = BatchNormalization(axis=3)(conv4)
#         relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_bias))
#     with tf.variable_scope('maxpool4'):
#         pool4 = tf.nn.max_pool(relu4,[1,2,2,1],[1,2,2,1],padding='VALID')
#     pool_shape = pool4.get_shape().as_list()
#     nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
#     reshape_pool = tf.layers.flatten(pool4)
#     with tf.variable_scope('fc1'):
#         fc1_weight = tf.get_variable('weight',[nodes,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
#         if regularizer != None:
#             tf.add_to_collection('losses',regularizer(fc1_weight))
#         fc1_bias = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.))
#         fc1 = tf.matmul(reshape_pool,fc1_weight)+fc1_bias
#         fc1 = BatchNormalization(axis=1)(fc1)
#         relufc1 = tf.nn.relu(fc1)
#         if train:
#             fc1 = tf.nn.dropout(fc1, keep_prob=1)
#     with tf.variable_scope('fc2'):
#         fc2_weight = tf.get_variable('weight',[64,2],initializer=tf.truncated_normal_initializer(stddev=0.1))
#         if regularizer != None:
#             tf.add_to_collection('losses',regularizer(fc2_weight))
#         fc2_bias = tf.get_variable('bias',[2],initializer=tf.constant_initializer(0.))
#         logit = tf.matmul(relufc1,fc2_weight)+fc2_bias
#     return logit

BATCH_SIZE = 8
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.9
REGULARIZATION_RATE = 0.0001
MAX_EPOCH = 100 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率
MODEL_SAVE_PATH = "./check"
MODEL_NAME = "model.ckpt"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def train(cat_dog):
    x = tf.placeholder(tf.float32,shape=(None,256,256,3),name='x-input')
    y_ = tf.placeholder(tf.float32,shape = (None,2),name='y_input')
   # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = cat_dog_CBAMmodel(x)
    global_step = tf.Variable(0,trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variabel_averages_op = variable_averages.apply(tf.trainable_variables())
    #loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean
    #设置指数衰减学习率
    Num_iters = int(cat_dog['X_train'][:].shape[0]/BATCH_SIZE) #全部训练数据计算一遍需要的迭代次数
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, #基础的学习率，随着迭代的进行，更新变量使用的学习率在此基础上递减
        global_step,#当前迭代轮数，喂入一次 BACTH_SIZE 计为一次 global_step
        Num_iters,#全部训练数据计算一遍需要的迭代次数
        LEARNING_RATE_DECAY)  #学习率衰减
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(MAX_EPOCH):
            for k in range(Num_iters):
                xs,ys = cat_dog["X_train"][(k)*BATCH_SIZE:(k + 1)*BATCH_SIZE] ,cat_dog["Y_train"][(k)*BATCH_SIZE:(k + 1)*BATCH_SIZE]
                xs = np.reshape(xs,[BATCH_SIZE,256,256,3])
                _,_,loss_value,step = sess.run([train_step,variabel_averages_op,loss,global_step],feed_dict={x:xs,y_:ys})
                if k %100 ==0:
                    print("Epoch {} || {} iterations ||loss on training batch is {}.".format(i+1,step%Num_iters,loss_value))
                #保存当前模型，文件名加上当前训练轮数
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)

data_dir = './train1/'
data_name = 'data.h5'
def main(argv=None):
    X_train,Y_train,X_test,Y_test = load_data(data_dir,data_name)
    Y_train = np.squeeze(one_hot_matrix(Y_train,2))
    cat_dog = {'X_train':X_train,
               'Y_train':Y_train}
    train(cat_dog)
if __name__=='__main__':
    tf.app.run()
