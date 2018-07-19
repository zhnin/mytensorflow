# @Time    : 2018/7/19 17:36
# @Author  : cap
# @FileName: mnist_acgan.py
# @Software: PyCharm Community Edition
# @introduction:
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
"""
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import numpy as np

np.random.seed(1337)
num_classes = 10


def build_generator(latent_size):
    cnn = Sequential()

    cnn.add(Dense(3 * 3 * 384, input_shape=latent_size, activation='relu'))
    cnn.add(Reshape((3, 3, 384)))

    cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid', activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh',
                            kernel_initializer='glorot_normal'))

    # this is the z space
    latent = Input(shape=(latent_size))

    # this will be the label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(num_classes, latent_size, embeddings_initializer='glorot_normal')(image_class))

    h = multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2, input_shape=(28, 28, 1)))

    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, 1, 'same'))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, 1, 'same'))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, 1, 'same'))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])


if __name__ == "__main__":
    epochs = 100
    batch_size = 100
    latent_size = 100

    # Adam paramters
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                          loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

    discriminator.summary()

    # build a generator
    generator = build_generator(latent_size)

    pass