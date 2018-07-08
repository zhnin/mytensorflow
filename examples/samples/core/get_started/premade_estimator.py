# @Time    : 2018/7/6 23:11
# @Author  : cap
# @FileName: premade_estimator.py
# @Software: PyCharm Community Edition
# @introduction:
import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='')
parser.add_argument('--train_step', default=1000, type=int, help='')


def main(_):
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_columns = []
    # train_x.keys()获取列名
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    print(my_feature_columns)
    # 创建有两个隐藏层，每层10个节点的神经网络
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3
    )

    # train
    classifier.train(
        input_fn=lambda :iris_data.train_input_fn(train_x, train_y, FLAGS.batch_size),
        steps=FLAGS.train_step
    )
    # evaluate
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y, FLAGS.batch_size)
    )
    #print('===========', eval_result)
    # {'accuracy': 0.96666664, 'average_loss': 0.065968044, 'loss': 1.9790413, 'global_step': 1000}
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # predict
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x, labels=None, batch_size=FLAGS.batch_size)
    )

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}"({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(iris_data.SPECIES[class_id], 100 * probability, expec))

if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
