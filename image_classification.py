import os
from PIL import Image

orig_dir,save_dir = './train' ,'./train1'
test_dir = './test'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(save_dir+'/cats') and os.path.exists(save_dir+'/dogs') :
    os.mkdir(save_dir +'/cats')
    os.mkdir(save_dir +'/dogs')
# cats如果之前存放的有文件，全部清除
for i in os.listdir(save_dir+'/cats'):
    path_file = os.path.join(save_dir+'/cats', i)
    if os.path.isfile(path_file):
        os.remove(path_file)
# dogs如果之前存放的有文件，全部清除
for i in os.listdir(save_dir+'/dogs'):
    path_file = os.path.join(save_dir+'/dogs', i)
    if os.path.isfile(path_file):
        os.remove(path_file)
# test如果之前存放的有文件，全部清除
for i in os.listdir(test_dir):
    path_file = os.path.join(test_dir, i)
    if os.path.isfile(path_file):
        os.remove(path_file)
img_cats = []
img_dogs = []
for img in os.listdir(orig_dir):
    img_name = img.split('.')[0]
    if img_name == 'cat':
        img_cats.append(img)
    elif img_name == 'dog':
        img_dogs.append(img)
#从中各选2000张用来训练和测试
for img_cat in img_cats[0:2000]:
        im = Image.open(orig_dir+'/'+img_cat)
        im.save(save_dir+'/cats'+'/'+img_cat)
        im.close()
for img_dog in img_dogs[0:2000]:
        im = Image.open(orig_dir+'/'+img_dog)
        im.save(save_dir+'/dogs'+'/'+img_dog)
        im.close()
for img_cat in img_cats[2000:2500]:
    im = Image.open(orig_dir + '/' + img_cat)
    im.save(test_dir +'/'+img_cat)
    im.close()
for img_dog in img_dogs[2000:2500]:
    im = Image.open(orig_dir + '/' + img_dog)
    im.save(test_dir +'/'+ img_dog)
    im.close()
