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

#使用yield制作数据集生成器
def data_generator(X_train,Y_train,Batch_size):
    n= len(X_train)
    i = 0
    while True:
        image_data = []
        label  = []
        for b in range(Batch_size):
            i%=n  #无限迭代的时候控制索引 不超过总样本
            image_data.append(X_train[i])
            label.append(Y_train[i])
            i += 1
        image_data = np.array(image_data)
        label = np.array(label)
        yield (image_data,label) #feed数据，每次batchsize个训练数据，无限循环，调用一次

def train(Cat_Dog,X_train,Y_train):
    Batch_size = 16
    val_split = 0.1
    num_val = int(len(X_train[:])*val_split)
    num_train = len(X_train[:]) - num_val
    Cat_Dog.compile(optimizer='Adam',#使用adam默认的参数训练，等于keras.optimizer.adam(lr=0.001)
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    # Cat_Dog.compile(optimizer=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    #                 loss='binary_crossentropy', metrics=['accuracy'])
    # Cat_Dog.summary()
    #开始训练，保存参数

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir +"ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)
    # 构建数据增强生成器
    # Data_Hence = image.ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    # 	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    # 	horizontal_flip=True, fill_mode="nearest")
    # Cat_Dog.fit_generator(Data_Hence.flow(X_train[:num_train],Y_train[:num_train],Batch_size),#.flow生成器方法
    #                       steps_per_epoch=max(1, num_train//Batch_size),
    #                       validation_data=(X_train[num_train:],Y_train[num_train:]),
    #                       validation_steps = max(1, num_val // Batch_size),
    #                       epochs=100,
    #                       callbacks=[logging, checkpoint, reduce_lr])
    Cat_Dog.fit_generator(data_generator(X_train[:num_train],Y_train[:num_train],Batch_size),#yield生成器方法
                          steps_per_epoch=max(1, num_train//Batch_size),
                          validation_data=data_generator(X_train[num_train:],Y_train[num_train:],Batch_size),
                          validation_steps = max(1, num_val // Batch_size),
                          epochs=10,
                          callbacks=[logging, checkpoint, reduce_lr])
    Cat_Dog.save('cat_dog.h5') #保存整个模型 结构和参数

if __name__ == '__main__':
    ###导入训练数据
    data_dir = './train1/'
    data_name = 'GoogLeNetData_224.h5'
    log_dir = './log/'
    X_train, Y_train, X_test, Y_test = load_data(data_dir, data_name)
    Y_train = np.squeeze(one_hot_matrix(Y_train, 2))
    Y_test = np.squeeze(one_hot_matrix(Y_test, 2))
    Cat_Dog = GoogLeNet(intensor=(224, 224, 3))
    train(Cat_Dog,X_train,Y_train)
    preds = Cat_Dog.evaluate(x=X_test, y=Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    #测试自己的图片
    img_path = './test/cat.1186.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(Cat_Dog.predict(x))