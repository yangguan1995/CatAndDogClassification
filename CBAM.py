import keras.backend as k
from keras.layers import  BatchNormalization,Input,Conv2D,AveragePooling2D,MaxPooling2D,Activation,Flatten,Dense,Reshape
from keras.models import Model
import tensorflow as tf
from keras.layers import Lambda,Concatenate,Add,Multiply


####在此加CBAM注意力机制！！！！！！！！！！！
def mul(A):
    return tf.multiply(A[0],A[1])
def reshape(A,shape):
    return tf.reshape(A,shape)
def reduce_mean(x):
    return tf.reduce_mean(input_tensor=x,axis=-1,keep_dims=True)
def reduce_max(x):
    return tf.reduce_max(input_tensor=x, axis=-1, keep_dims=True)
def get_shape(input_tensor):
    shape = input_tensor.get_shape().as_list()
    return shape

scale=8
input_tensor = Input((28,28,128))
shape = input_tensor.get_shape().as_list()
"""通道注意力机制"""
##avgpooling channel
avg = AveragePooling2D(pool_size=(shape[1],shape[2]),strides=(1,1))(input_tensor)
avg = Flatten()(avg)
FC_avg = Dense(int(shape[3]/scale))(avg)
FC_avg = BatchNormalization()(FC_avg)
FC_avg = Activation('relu')(FC_avg)
FC2_avg = Dense(shape[3])(FC_avg)
##maxpooling channel
max = MaxPooling2D(pool_size=(shape[1],shape[2]),strides=(1,1))(input_tensor)
# max = tf.nn.max_pool(value=input, ksize=poolsize, strides=[1,1,1,1], padding='VALID')
max = Flatten()(max)
FC_max = Dense(int(shape[3]/scale))(max)
FC_max = BatchNormalization()(FC_max)
FC_max = Activation('relu')(FC_max)
FC2_max = Dense(shape[3])(FC_max)
AD = Add()([FC2_avg,FC2_max])
# add avg and max channel
logit = Activation(activation='sigmoid')(AD)
mask1 = Reshape([1,1,shape[3]])(logit)
outputs1 = Multiply()([input_tensor,mask1]) #[28,28,128]
"""空间注意力机制"""
avg = Lambda(reduce_mean)(outputs1)
max = Lambda(reduce_max)(outputs1)
add_max_avg = Concatenate()([avg,max])
add_max_avg = Reshape([shape[1],shape[2],2])(add_max_avg)
sig = Conv2D(1,(7,7),padding='same')(add_max_avg)
sig = Activation('sigmoid')(sig)
mask2 = Reshape([shape[1],shape[2], 1])(sig)
outputs2 = Multiply()([outputs1,mask2])
model = Model(inputs=input_tensor, outputs=outputs2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.summary()


