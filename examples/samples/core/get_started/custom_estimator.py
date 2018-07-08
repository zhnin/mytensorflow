# @Time    : 2018/7/7 13:52
# @Author  : cap
# @FileName: custom_estimator.py
# @Software: PyCharm Community Edition
# @introduction:
import argparse
import tensorflow as tf
import iris_data


def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    # logit
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # prediction
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
     # accuracy
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # trian op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(_):
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # classifier
    classifier = tf.estimator.Estimator(model_fn=my_model,
                                        params={
                                            'feature_columns': my_feature_columns,
                                            'hidden_units': [10, 10],
                                            'n_classes': 3
                                        })

    classifier.train(input_fn=lambda :iris_data.train_input_fn(train_x, train_y, FLAGS.batch_size),
                     steps=FLAGS.train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda :iris_data.eval_input_fn(test_x, test_y, FLAGS.batch_size)
    )
    print('\nTest set accuracy:{accuracy:0.3f}\n'.format(**eval_result))

    # predict
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda :iris_data.eval_input_fn(predict_x, None, FLAGS.batch_size)
    )

    for pred_dict, expec in zip(predictions, expected):
        template = '\nPrediction is "{} ({:.1f}%), expected {}"'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id], 100 * probability, expec))

class MyArgParse(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--batch_size', type=int, default=100, help='')
        self.add_argument('--train_steps', type=int, default=1000, help='')


if __name__ == '__main__':
    parser = MyArgParse()
    FLAGS, _ = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
