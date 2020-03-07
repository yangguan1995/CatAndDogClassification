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
#读取数据集

data_dir = './train1/'
data_name = 'GoogLeNetData_224.h5'
X_train, Y_train, X_test, Y_test = load_data(data_dir,data_name)
#当数据为多分类>2时，使用onehot编码，不能使用binary_crossentropy
Y_train = np.squeeze(one_hot_matrix(Y_train,2))
Y_test = np.squeeze(one_hot_matrix(Y_test,2))
####写一个Inceptioon的函数,在使用keras.Model类时，要将其改用Lambda层使用
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

def Cat_Dog_GoogLeNet(input_shape):
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    X_input = Input(input_shape)  # placeholder
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
    X = Lambda(InceptionBlock_Google,arguments={'c1':64,'c2':(96,128),'c3':(16,32),'c4':32 })(X)
    X = Lambda(InceptionBlock_Google,arguments={'c1':128,'c2':(128,192),'c3':(32,96),'c4':64})(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X)
    #Block4
    X = Lambda(InceptionBlock_Google,arguments={'c1':192,'c2':(96,208),'c3':(16,48),'c4':64 })(X)
    X = Lambda(InceptionBlock_Google,arguments={'c1':160,'c2':(112,224),'c3':(24,64),'c4':64 })(X)
    X = Lambda(InceptionBlock_Google,arguments={'c1':128,'c2':(128,256),'c3':(24,64),'c4':64 })(X)
    X = Lambda(InceptionBlock_Google,arguments={'c1':112,'c2':(144,288),'c3':(32,64),'c4':64 })(X)
    X = Lambda(InceptionBlock_Google,arguments={'c1':256,'c2':(160,320),'c3':(32,128),'c4':128 })(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X)
    #Block5
    X = Lambda(InceptionBlock_Google,arguments={'c1':256,'c2':(160,320),'c3':(32,128),'c4':128 })(X)
    X = Lambda(InceptionBlock_Google,arguments={'c1':384,'c2':(192,384),'c3':(48,128),'c4':128 })(X)
    X = AveragePooling2D(pool_size=(7,7),strides=(1,1))(X)
    # FC
    X = Flatten()(X)
    X = Dense(64)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)
    Y = Dense(2, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=Y, name='Cat_Dog_GoogLeNet')
    return model



log_dir = './log/'
Cat_Dog = Cat_Dog_GoogLeNet(input_shape=(224, 224, 3))
Cat_Dog.compile(optimizer=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                loss='binary_crossentropy', metrics=['accuracy'])
Cat_Dog.summary()
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir +"ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)
# 对于简单的数据集，fit函数够用
Cat_Dog.fit(x=X_train, y=Y_train, batch_size=32, epochs=50, validation_split=0.2,callbacks=[logging, checkpoint, reduce_lr])
Cat_Dog.save('cat_dog.model')
preds = Cat_Dog.evaluate(x=X_test, y=Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
##测试自己的图片
img_path = './test/cat.1186.jpg'
img = image.load_img(img_path, target_size=(224, 224))
imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(Cat_Dog.predict(x))