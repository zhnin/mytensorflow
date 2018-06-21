# @Time    : 2018/6/21 14:54
# @Author  : cap
# @FileName: myMnist.py
# @Software: PyCharm Community Edition
# @introduction:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义各层的节点数，这里只使用一个隐藏层
INPUT_NODES = 28 * 28
OUTPUT_NODES = 10
HIDDEN_NODES = 500

BATCH_SIZE = 100

# 定义滑动平均参数
MOVING_AVERAGE_DECAY = 0.99

# 定义正则化参数
REGULARIZATION_RATE = 0.0001

# 定义学习率优化参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

# 训练次数
TRAINING_STEPS = 10000


def getFinalOutput(input_tensor, weights1, biases1, weights2, biases2, avg_class=None):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    # 定义输入值和输出值
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODES), name='input')
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODES), name='output')

    # 初始化权重和偏移量
    weights1 = tf.Variable(tf.truncated_normal((INPUT_NODES, HIDDEN_NODES), stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=(HIDDEN_NODES,)))

    weights2 = tf.Variable(tf.truncated_normal((HIDDEN_NODES, OUTPUT_NODES), stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=(OUTPUT_NODES,)))

    # 定义一个没有滑动平均的输出值
    output = getFinalOutput(x, weights1, biases1, weights2, biases2)

    # 创建滑动平均类
    GLOBAL_STEP = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, GLOBAL_STEP)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_output = getFinalOutput(x, weights1, biases1, weights2, biases2, variable_averages)

    # 计算损失函数交叉熵+softmax
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 正则项
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    # 最终损失函数
    loss = cross_entropy_mean + regularization

    # 定义衰减学习率(指数衰减法)
    LEARNING_RATE = tf.train.exponential_decay(LEARNING_RATE_BASE, GLOBAL_STEP, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    # 梯度下降更新权重值
    gdo = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_step = gdo.minimize(loss, global_step=GLOBAL_STEP)

    # 打包更新操作，这里有两个动作，train_step,variable_averages_op
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 输出正确率
    correct_predict = tf.equal(tf.argmax(average_output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    # 初始会话
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, validate_feed)
                print('After %d training steps, validation accuracy using average model is %g' % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, test_feed)
        print('After %d training steps, test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('D:/softfiles/workspace/tensorflow/data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
