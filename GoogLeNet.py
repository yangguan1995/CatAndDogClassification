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