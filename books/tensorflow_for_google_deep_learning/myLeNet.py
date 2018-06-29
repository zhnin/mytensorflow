# @Time    : 2018/6/29 15:00
# @Author  : cap
# @FileName: myLeNet.py
# @Software: PyCharm Community Edition
# @introduction:
import tensorflow as tf


BATCH_SIZE = 100

INPUT_SIZE = 28
NUM_CHANNEL = 3

CONV1_SIZE = 5
CONV1_DEEP = 32

CONV2_SIZE = 5
CONV2_DEEP = 64

FC_SIZE = 512
NUM_LABELS = 10


def inference(input_tensor, train, regularizer):
    # 第一卷积层的输出为28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))

        biases = tf.get_variable('biases', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, [1, 1, 1, 1], 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))

    # 第二层池化，最大池化，输入是28*28*32 输出是14*14*32，大小为2*2 步长为2
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # 第三层卷积层，输入为14*14*32，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weight, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))

    # 第四层为池化层，输入为14*14*64 ,输出为7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # 第五层全连接层输入为7*7*64转为标准输出格式，batch*input_nodes
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    # 第五层的输出为512
    with tf.variable_scope('layer5-fc1'):
        fc1_weight = tf.get_variable('weight', [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weight))

        fc1_biases = tf.get_variable('biases', [FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_biases)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层全连接输入为512输出为10，最后进行softmax
    with tf.variable_scope('layer6-fc1'):
        fc2_weight = tf.get_variable('weight', [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weight))

        logit = tf.matmul(fc1, fc2_weight) + fc2_biases

    return logit


'''
# train中做以下修改，
def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHANNEL], name='x-input')
    
    reshaped_xs = np.reshape(xs, (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHANNEL))
'''
