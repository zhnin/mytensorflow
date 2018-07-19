# @Time    : 2018/7/18 19:42
# @Author  : cap
# @FileName: cifar10_cnn_capsule.py
# @Software: PyCharm Community Edition
# @introduction:
import os

import keras
from  keras import backend as K
from keras.layers import Layer
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras import activations
from keras.layers import Lambda
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import input_data


# activation
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, marge = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - marge - y_pred)) + lamb * (1 - y_true) * K.square(K.relu(y_pred - marge)), axis=-1)


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(name='capsule_kernel',
                                          shape=(1, input_dim_capsule, self.num_capsule * self.dim_capsule),
                                          initializer='glorot_uniform',
                                          trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernal = self.add_weight(name='capsule_kernel',
                                          shape=(input_num_capsule, input_dim_capsule, self.num_capsule * self.dim_capsule),
                                          initializer='glorot_uniform',
                                          trainable=True)

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                                      (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


batch_size = 128
num_classes = 10
epochs = 20
data_augmentation = True
num_predictions = 20
save_dir = 'D:\\softfiles\\workspace\\data\\tensorflow\\model\\cifar10\\cifar10_cnn_capsule'
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = input_data.input_data('cifar10')

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Conv2d
input_image = Input(shape=(None, None, 3))
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D()(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)

x = keras.layers.Reshape((-1, 128))(x)
capsule = Capsule(10, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)

model.compile(optimizer='adam', loss=margin_loss, metrics=['acc'])

model.summary()

if not data_augmentation:
    print('Not using the data augmenttation.')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
else:
    print('Using real-time data augmentation')
    datagen = ImageDataGenerator()
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accu:', score[1])

