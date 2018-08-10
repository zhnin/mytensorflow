# @Time    : 2018/8/9 19:28
# @Author  : cap
# @FileName: cifar10_resnet_v1_v2_test.py
# @Software: PyCharm Community Edition
# @introduction:
from keras.models import load_model
import numpy as np
from PIL import Image

# 加载本地图片
path = r'D:\softfiles\workspace\git\mypython\ml\data\objects\testing\airplane\0020.jpg'
im = Image.open(path).resize((32,32))
im = np.asarray(im)
x_test = im.astype('float32') / 255
x_test = np.expand_dims(x_test, 0)

# decode label for man
def get_label(args):
    import pickle
    labels = r'D:\softfiles\workspace\data\tensorflow\data\cifar10\cifar-10-batches-py\batches.meta'
    with open(labels, 'rb') as f:
        label_list = pickle.load(f, encoding='utf-8')['label_names']
        print(label_list)
    return label_list[args]

# 加载训练好的cifar10.h5模型
save_dir = r'D:\softfiles\workspace\data\tensorflow\model\cifar10\resnet\cifar10_ResNet20v1_model.094.h5'
model = load_model(save_dir)
# model.summary()
result = model.predict(x_test)
predict = get_label(np.argmax(result))
print(predict)