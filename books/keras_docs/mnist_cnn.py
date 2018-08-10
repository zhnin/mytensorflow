# @Time    : 2018/7/18 15:29
# @Author  : cap
# @FileName: mnist_cnn.py
# @Software: PyCharm Community Edition
# @introduction: cnn
"""
Test loss: 0.024420398353816063
Test accu: 0.9924
"""
import os
import keras
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import input_data


batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = input_data.input_data('mnist')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# create models conv2d(32) - conv2d(64) - maxpool - dropout - floatten - dense(128) - dropout - dense(10)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
save_dir = r'D:\softfiles\workspace\data\tensorflow\model\mnist\cnn'

model_name = 'mnist_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

callbacks = [checkpoint]

model.compile(optimizer=keras.optimizers.Adadelta(),
              loss='categorical_crossentropy',
              metrics=['acc'])

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accu:', score[1])