# @Time    : 2018/6/22 0:39
# @Author  : cap
# @FileName: mnist_eval.py
# @Software: PyCharm Community Edition
# @introduction:
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train


EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, (None, mnist_inference.INPUT_NODES), name='x')
        y_ = tf.placeholder(tf.float32, (None, mnist_inference.OUTPUT_NODES), name='y')

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training steps, validation accuracy=%g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('D:/softfiles/workspace/tensorflow/data/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
