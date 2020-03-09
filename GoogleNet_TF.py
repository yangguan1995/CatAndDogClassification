#使用TF加keras搭建一个简单的卷积神经网络
import  tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras.layers import Input ,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import  MaxPooling2D, Dropout,AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from utils import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
####写一个Inceptioon的函数,在使用keras.Model类时，要将其改用Lambda层使用

tf.reset_default_graph()
def InceptionBlock_Google(input_tensor,c1,c2,c3,c4):
    x1 = Conv2D(filters=c1,kernel_size=(1,1))(input_tensor)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2_1 = Conv2D(filters=c2[0],kernel_size=(1,1))(input_tensor)
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_1 = Activation('relu')(x2_1)
    x2_2 = Conv2D(filters=c2[1],kernel_size=(3,3),padding='same')(x2_1)
    x2_2 = BatchNormalization(axis=3)(x2_2)
    x2_2 = Activation('relu')(x2_2)

    x3_1 = Conv2D(filters=c3[0],kernel_size=(1,1))(input_tensor)
    x3_1 = BatchNormalization(axis=3)(x3_1)
    x3_1 = Activation('relu')(x3_1)
    x3_2 = Conv2D(filters=c3[1],kernel_size=(5,5),padding='same')(x3_1)
    x3_2 = BatchNormalization(axis=3)(x3_2)
    x3_2 = Activation('relu')(x3_2)

    x4_1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_tensor)
    x4_2 = Conv2D(filters=c4,kernel_size=(1,1))(x4_1)
    x4_2 = BatchNormalization(axis=3)(x4_2)
    x4_2 = Activation('relu')(x4_2)
    return  k.concatenate([x1,x2_2,x3_2,x4_2],axis=3)

def GoogLeNet(X_input):

    # X_input = Input(input_shape)  # placeholder
    #Block1
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)
    #Block2
    X = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(192,kernel_size=(3,3),padding='same',activation='relu')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)
    #Block3
    X = InceptionBlock_Google(X,64,(96,128),(16,32),32)
    X = InceptionBlock_Google(X,128,(128,192),(32,96),64)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X)
    #Block4
    X = InceptionBlock_Google(X,192,(96,208),(16,48),64)
    X = InceptionBlock_Google(X,160,(112,224),(24,64),64)
    X = InceptionBlock_Google(X,128,(128,256),(24,64),64)
    X = InceptionBlock_Google(X,112,(144,288),(32,64),64)
    X = InceptionBlock_Google(X,256,(160,320),(32,128),128)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X)
    #Block5
    X = InceptionBlock_Google(X,256,(160,320),(32,128),128)
    X = InceptionBlock_Google(X,384,(192,384),(48,128),28)
    X = AveragePooling2D(pool_size=(7,7),strides=(1,1))(X)
    # FC
    X = Flatten()(X)
    # X = Dense(64)(X)
    # X = BatchNormalization()(X)
    # X = Activation('relu')(X)
    # X = Dropout(0.4)(X)
    X = Dense(2)(X)
    Y = BatchNormalization(axis=1)(X)
    return Y


BATCH_SIZE = 8
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.9
MAX_EPOCH = 50 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率
MODEL_SAVE_PATH = "./tf_results"
MODEL_NAME = "model.ckpt"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def train(cat_dog):
    x = tf.placeholder(tf.float32,shape=(None,224,224,3),name='x-input')
    y_ = tf.placeholder(tf.float32,shape = (None,2),name='y_input')
   # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = GoogLeNet(x)
    global_step = tf.Variable(0,trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variabel_averages_op = variable_averages.apply(tf.trainable_variables())
    #loss
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_) #独立不互斥分类，一图多目标
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)#独立互斥分类，一图一目标，都要onehot
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))#独立互斥分类，一图一目标，不要onehot
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
                xs = np.reshape(xs,[BATCH_SIZE,224,224,3])
                _,_,loss_value,step = sess.run([train_step,variabel_averages_op,loss,global_step],feed_dict={x:xs,y_:ys})
                if k %100 ==0:
                    print("Epoch {} || {} iterations ||loss on training batch is {}.".format(i+1,step%Num_iters,loss_value))
                #保存当前模型，文件名加上当前训练轮数
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)

data_dir = './train1/'
data_name = 'GoogLeNetData_224.h5'
def main(argv=None):
    X_train,Y_train,X_test,Y_test = load_data(data_dir,data_name)
    Y_train = np.squeeze(one_hot_matrix(Y_train,2))
    cat_dog = {'X_train':X_train,
               'Y_train':Y_train}
    train(cat_dog)
if __name__=='__main__':
    tf.app.run()
