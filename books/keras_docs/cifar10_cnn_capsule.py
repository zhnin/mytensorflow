# @Time    : 2018/7/18 19:42
# @Author  : cap
# @FileName: cifar10_cnn_capsule.py
# @Software: PyCharm Community Edition
# @introduction:
# from  keras import backend as K
import numpy as np

a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.exp(a - np.max(a, 1, keepdims=True))

c = np.square(a)
d = np.sum(c, -1, keepdims=True)
e = np.sqrt(d / (0.5 + d))
print(e * a)
print(b / np.sum(b,-1,keepdims=True))
