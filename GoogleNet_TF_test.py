
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
from GoogleNet_TF import GoogLeNet
model_path = './log/ep048-loss0.302-val_loss0.408.h5'

img_path = './test/dog.11811.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x_im = image.img_to_array(img)
x_im = np.expand_dims(x_im, axis=0)
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x-input-test')
y = GoogLeNet(x)
test_feed = {x: x_im}
saver = tf.train.Saver()
with tf.Session() as sess:
    model_path = ''
    saver.restore(sess, model_path)  # 在当前sess下恢复ckpt中所有变量
    result = sess.run(y,feed_dict=x_im)
    print('这个动物属于类：{}'.format(result))
plt.imshow(img)
plt.show()