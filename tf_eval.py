import tensorflow as tf
import os
import numpy as np
from PIL import Image
from utils import *
import use_tf_conv
from use_tf_conv import cat_dog_model

tf.reset_default_graph()
def model_eval(cat_dog_test):
    x = tf.placeholder(tf.float32,shape = [None,256,256,3],name = 'x-input-test')
    y_ = tf.placeholder(tf.float32,shape = [None,2],name = 'y-test-pred')
    test_feed = {x:cat_dog_test['X_test'][0:10],y_:cat_dog_test['Y_test'][0:10]}
    y = cat_dog_model(x)
    correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。这样就可以完全
    # 共用mnist_inference.py中定义的前向传播过程
    variable_average = tf.train.ExponentialMovingAverage(use_tf_conv.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(use_tf_conv.MODEL_SAVE_PATH)
        #ckpt返回的是                  model_checkpoint_path:"Check/model.ckpt-20000"
        # 以及所有的ckpt文件           all_model_checkpoint_paths:.......
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)  #在当前sess下恢复ckpt中所有变量
            accuracy_score = sess.run(accuracy,feed_dict=test_feed)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("After {} training steps , test accuracy = {}".format(global_step,accuracy_score))

def main(argv = None):
    _,_,X_test,Y_test = load_data()
    Y_test = np.squeeze(one_hot_matrix(Y_test,2))
    cat_dog_test = {'X_test':X_test,
                    'Y_test':Y_test}
    model_eval(cat_dog_test)


if __name__ == '__main__':
    tf.app.run()

