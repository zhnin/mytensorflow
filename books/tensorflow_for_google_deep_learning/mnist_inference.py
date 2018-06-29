# @Time    : 2018/6/21 23:32
# @Author  : cap
# @FileName: mnist_inference.py
# @Software: PyCharm Community Edition
# @introduction:
import tensorflow as tf

INPUT_NODES = 28 * 28
OUTPUT_NODES = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer=None):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义计算图，regularizere为正则类，这里不对偏执项进行过拟合操作
def inference(input_tensor, regularizer=None):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable((INPUT_NODES, LAYER1_NODE), regularizer)
        biases = tf.get_variable('biases', (LAYER1_NODE,), initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)) + biases

    with tf.variable_scope('layer2'):
        weights = get_weight_variable((LAYER1_NODE, OUTPUT_NODES), regularizer)
        biases = tf.get_variable('biases', (OUTPUT_NODES,), initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
