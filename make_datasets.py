#以猫狗分类为例，将两类图放在两个文件夹中
import os
import numpy as np
from PIL import Image
import h5py

HDF5_DISABLE_VERSION_CHECK = 1
def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir + '/cats'):
        cats.append(file_dir + '/cats' + '/' + file)
        label_cats.append(0)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    for file in os.listdir(file_dir + '/dogs'):
        dogs.append(file_dir + '/dogs' + '/' + file)
        label_dogs.append(1)

    # 把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list
 # 返回两个list 分别为图片文件名及其标签  顺序已被打乱

#指定图片地址
train_dir = './train1'
image_list,label_list = get_files(train_dir)
print('训练集共有{}张图片'.format(len(image_list)))

lentrain = int(np.floor(0.8*len(image_list)))
lentest = int(len(image_list)-lentrain)
Train_image =  np.random.rand(lentrain, 224, 224, 3).astype('float32')  #由256->224
Train_label = np.random.rand(lentrain, 1).astype('float32')
Test_image =  np.random.rand(lentest, 224, 224, 3).astype('float32')
Test_label = np.random.rand(lentest, 1).astype('float32')
# #把训练集图片读入并保存为数组用于训练
for i in range(lentrain):
    im = Image.open(image_list[i])
    im = im.resize((224, 224),Image.BICUBIC)
    Train_image[i] = im;
    Train_label[i] = np.array(label_list[i])
    
        
for i in range(lentrain, len(image_list)):
    im = Image.open(image_list[i])
    im = im.resize((224, 224),Image.BICUBIC)
    Test_image[i-lentrain] = im
    Test_label[i-lentrain] = np.array(label_list[i])
# Create a new file
f = h5py.File(os.path.join(train_dir,'GoogLeNetData_224.h5'), 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()

