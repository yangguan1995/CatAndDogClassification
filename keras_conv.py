#使用TF加keras搭建一个简单的卷积神经网络
import  tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras.layers import Input ,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import  MaxPooling2D, Dropout
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
data_name = 'data.h5'
X_train, Y_train, X_test, Y_test = load_data(data_dir,data_name)
# Normalize image vectors.
# Reshape
def Cat_Dog_Model(input_shape):
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    X_input = Input(input_shape)  #placeholder

    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding = 'same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)
    
    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1),padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    # FC
    X = Flatten()(X)
    X = Dense(64)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    Y = Dense(1,activation='sigmoid')(X)
    

    model = Model(inputs=X_input, outputs=Y, name='Cat_Dog_Model')
    return model
log_dir = './check'
Cat_Dog = Cat_Dog_Model(input_shape=(256,256,3))
Cat_Dog.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,min_lr=1e-10, verbose=1)
#对于简单的数据集，fit函数够用
Cat_Dog.fit(x=X_train, y=Y_train, batch_size=10, epochs=20,callbacks=[logging,checkpoint,reduce_lr])
preds = Cat_Dog.evaluate(x=X_test, y=Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
##测试自己的图片
img_path = './test/dog.3435.jpg'
img = image.load_img(img_path, target_size=(256, 256))
imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(Cat_Dog.predict(x))