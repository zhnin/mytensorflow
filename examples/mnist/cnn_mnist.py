# @Time    : 2018/7/7 15:16
# @Author  : cap
# @FileName: cnn_mnist.py
# @Software: PyCharm Community Edition
# @introduction:
"""
1. conv1, `weights = [-1, 5, 5, 32]`, with ReLU
2. pool1, `kernel_size=[1, 2, 2, 1]`, `stride=[1, 2, 2, 1]`
3. conv2, `weights=[-1, 5, 5, 64]`, with ReLU
4. pool2, `kernel_size=[1, 2, 2, 1]`, `stride=[1, 2, 2, 1]`
5. dense1, `[-1, 1024]`, `dropout(0.4)`,
6. dense2, `[-1, 10]`
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    predictions = {
        'classes': tf.argmax(logits, 1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, 1), logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, train_op=train_op, loss=loss)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(_):
    mnist = input_data.read_data_sets('D:\\softfiles\\workspace\\data\\tensorflow\\data', one_hot=True)
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir='D:\\softfiles\\workspace\\data\\tensorflow\\model\\cnn_mnist')

    tensor_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': mnist.train.images},
        y=mnist.train.labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.validation.images},
        y=mnist.validation.labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
