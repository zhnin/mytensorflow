# @Time    : 2018/7/6 15:57
# @Author  : cap
# @FileName: mnist_estimator.py
# @Software: PyCharm Community Edition
# @introduction:

import argparse
import os
import tensorflow as tf


class Model(object):
    """"""
    def __init__(self, data_format):
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, 28, 28, 1]

        # 定义模型
        # con
        self.conv1 = tf.layers.Conv2D(32, 5, padding='same',
                                      data_format=data_format,
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(64, 5, padding='same',
                                      data_format=data_format,
                                      activation=tf.nn.relu)
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.4)
        self.max_pool2d = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

    def __call__(self, inputs, training):
        y = tf.reshape(inputs, self._input_shape)
        y = self.conv1(y)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = tf.layers.flatten(y)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        return self.fc2(y)


def model_fn(features, labels, mode, params):
    """参数为固定格式
    * `features`: This is the first item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `Tensor` or `dict` of same.
          * `labels`: This is the second item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `Tensor` or `dict` of same (for multi-head models). If
                 mode is `ModeKeys.PREDICT`, `labels=None` will be passed. If
                 the `model_fn`'s signature does not accept `mode`, the
                 `model_fn` must still be able to handle `labels=None`.
          * `mode`: Optional. Specifies if this training, evaluation or
                 prediction. See `ModeKeys`.
          * `params`: Optional `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config`: Optional configuration object. Will receive what is passed
                 to Estimator in `config` parameter, or the default `config`.
                 Allows updating things in your model_fn based on configuration
                 such as `num_ps_replicas`, or `model_dir`.
    """
    model = Model(params['data_format'])
    image = features
    # feature也可以为字典格式
    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
        # 如果为
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, 1),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            }
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 如果为训练，定义优化器，Logit， loss, accuracy
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1))

        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':tf.metrics.accuracy(labels=labels,predictions=tf.argmax(logits, 1))
            }
        )


def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of
  available GPUs.

  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.
  """
  from tensorflow.python.client import device_lib

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
      'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
      'must be a multiple of the number of available GPUs. '
      'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
      ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)



def decode_image(image):
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [])
    return tf.to_int32(label)

def data_set(images_file, labels_file):
    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16
    ).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8
    ).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def train(directory):
    images_file = os.path.join(directory, 'train-images-idx3-ubyte')
    labels_file = os.path.join(directory, 'train-labels-idx1-ubyte')
    return data_set(images_file, labels_file)

def test(directory):
    images_file = os.path.join(directory, 't10k-images-idx3-ubyte')
    labels_file = os.path.join(directory, 't10k-labels-idx1-ubyte')
    return data_set(images_file, labels_file)

def main(_):
    model_function = model_fn

    if FLAGS.multi_gpu:
        validate_batch_size_for_multi_gpu(FLAGS.batch_size)
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_fn, loss_reduction=tf.losses.Reduction.MEAN
        )

    data_format = FLAGS.data_format
    if data_format is None:
        data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=FLAGS.model_dir,
        params={
            'data_format': data_format,
            'multi_gpu': FLAGS.multi_gpu
        }
    )

    def train_input_fn():
        ds = train(FLAGS.data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size).repeat(FLAGS.train_epochs)
        return ds
    print(train_input_fn())
    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100
    )
    mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    def eval_input_fn():
        return test(FLAGS.data_dir).batch(FLAGS.batch_size).make_one_shot_iterator().get_next()

    eval_result = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print()
    print('Evaluation results:\n\t%s' % eval_result)

    if FLAGS.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'image': image,})
        mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)


class MNISTArgParser(argparse.ArgumentParser):
    """设置变量"""
    def __init__(self):
        super(MNISTArgParser, self).__init__()

        self.add_argument(
            '--multi_gpu',
            action='store_true',
            help='multi gpu'
        )
        self.add_argument(
            '--batch_size',
            type=int,
            default=100,
            help='batch size'
        )
        self.add_argument(
            '--data_dir',
            type=str,
            default='D:/softfiles/workspace/data/tensorflow/data/mnist_data',
            help='data dir'
        )
        self.add_argument(
            '--model_dir',
            type=str,
            default='D:/softfiles/workspace/data/tensorflow/data/mnist_model',
            help='model dir'
        )
        self.add_argument(
            '--train_epochs',
            type=int,
            default=20,
            help='epochs'
        )
        self.add_argument(
            '--data_format',
            type=str,
            default=None,
            choices=['channels_first', 'channels_last'],
            help=''
        )
        self.add_argument(
            '--export_dir',
            type=str,
            help=''
        )


if __name__ == '__main__':
    parser = MNISTArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)