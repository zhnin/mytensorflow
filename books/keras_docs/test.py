from PIL import Image
import numpy as np
path = r'D:\softfiles\workspace\git\mypython\ml\data\objects\testing\airplane\0020.jpg'
im = Image.open(path).resize((32,32))
im = np.asarray(im)
print(im.dtype)