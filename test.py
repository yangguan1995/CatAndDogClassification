#使用TF加keras搭建一个简单的卷积神经网络

import os
import tensorflow as tf
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
#读取数据集

data_dir = './train1/'
data_name = 'GoogLeNetData_224.h5'
X_train, Y_train, X_test, Y_test = load_data(data_dir,data_name)
#当数据为多分类>2时，使用onehot编码，不能使用binary_crossentropy
Y_train = np.squeeze(one_hot_matrix(Y_train,2))
Y_test = np.squeeze(one_hot_matrix(Y_test,2))
def concat(x):
    return k.concatenate(x,axis=3)
def InceptionBlock_Google(input_tensor,c1,c2,c3,c4):
    x1_1 = Conv2D(filters=c1,kernel_size=(1,1))(input_tensor)
    x1_2 = BatchNormalization(axis=3)(x1_1)
    x1_3 = Activation('relu')(x1_2)

    x2_1 = Conv2D(filters=c2[0],kernel_size=(1,1))(input_tensor)
    x2_2 = BatchNormalization(axis=3)(x2_1)
    x2_3 = Activation('relu')(x2_2)
    x2_4 = Conv2D(filters=c2[1],kernel_size=(3,3),padding='same')(x2_3)
    x2_5 = BatchNormalization(axis=3)(x2_4)
    x2_6 = Activation('relu')(x2_5)

    x3_1 = Conv2D(filters=c3[0],kernel_size=(1,1))(input_tensor)
    x3_2 = BatchNormalization(axis=3)(x3_1)
    x3_3 = Activation('relu')(x3_2)
    x3_4 = Conv2D(filters=c3[1],kernel_size=(5,5),padding='same')(x3_3)
    x3_5 = BatchNormalization(axis=3)(x3_4)
    x3_6 = Activation('relu')(x3_5)

    x4_1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_tensor)
    x4_2 = Conv2D(filters=c4,kernel_size=(1,1))(x4_1)
    x4_3 = BatchNormalization(axis=3)(x4_2)
    x4_4 = Activation('relu')(x4_3)
    Y = Lambda(concat)([x1_3,x2_6,x3_6,x4_4])
    return  Y  #返回Lambda层，可直接参与model的构建，符合keras的基本操作单元是层！

def GoogLeNet(intensor):
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    X_input = Input(intensor)  # placeholder
    #Block1
    X1_1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same')(X_input)
    X1_2 = BatchNormalization(axis=3)(X1_1)
    X1_3 = Activation('relu')(X1_2)
    X1_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X1_3)
    #Block2
    X2_1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(X1_4)
    X2_2 = BatchNormalization(axis=3)(X2_1)
    X2_3 = Activation('relu')(X2_2)
    X2_4 = Conv2D(192,kernel_size=(3,3),padding='same',activation='relu')(X2_3)
    X2_5 = BatchNormalization(axis=3)(X2_4)
    X2_6 = Activation('relu')(X2_5)
    X2_7 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X2_6)
    #Block3
    X3_1 = InceptionBlock_Google(X2_7,64,(96,128),(16,32),32)
    X3_2 = InceptionBlock_Google(X3_1,128,(128,192),(32,96),64)
    X3_3 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X3_2)
    #Block4
    X4_1 = InceptionBlock_Google(X3_3,192,(96,208),(16,48),64)
    X4_2 = InceptionBlock_Google(X4_1,160,(112,224),(24,64),64)
    X4_3 = InceptionBlock_Google(X4_2,128,(128,256),(24,64),64)
    X4_4 = InceptionBlock_Google(X4_3,112,(144,288),(32,64),64)
    X4_5 = InceptionBlock_Google(X4_4,256,(160,320),(32,128),128)
    X4_6 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X4_5)
    #Block5
    X5_1 = InceptionBlock_Google(X4_6,256,(160,320),(32,128),128)
    X5_2 = InceptionBlock_Google(X5_1,384,(192,384),(48,128),28)
    X5_3 = AveragePooling2D(pool_size=(7,7),strides=(1,1))(X5_2)
    # FC
    X6_1 = Flatten()(X5_3)
    X6_2= Dense(64)(X6_1)
    X6_3 = BatchNormalization()(X6_2)
    X6_4 = Activation('relu')(X6_3)
    X6_5 = Dropout(0.4)(X6_4)
    Yout = Dense(2, activation='sigmoid')(X6_5)
    model = Model(inputs = X_input,outputs = Yout)
    return model

model = GoogLeNet(intensor = (224,224,3))
model_path = './log/ep048-loss0.302-val_loss0.408.h5'
model_path = os.path.expanduser(model_path)
model.load_weights(model_path,by_name=True)
img_path = './test/cat.1186.jpg'
img = image.load_img(img_path, target_size=(224, 224))
imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(model.predict(x))
