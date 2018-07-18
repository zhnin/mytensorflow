# @Time    : 2018/7/18 14:45
# @Author  : cap
# @FileName: mnist_mlp.py
# @Software: PyCharm Community Edition
# @introduction: DNN
"""
Test loss: 0.12962960983521643
Test accu: 0.9825
"""
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np


batch_size = 128
num_classes = 10
epochs = 20

## read the data from local data
path = "D:\\softfiles\\workspace\\data\\tensorflow\\data\\mnist_data\\mnist.npz"
with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# data preprocessing
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 将[0-9]的label表现形式，转为num_classes列用1标识的形式[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# create model, dense,relu - dropout - dense,relu - dropout - dense,softmax - output
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# print summary
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accu:', score[1])