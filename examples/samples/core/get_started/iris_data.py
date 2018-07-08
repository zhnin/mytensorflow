# @Time    : 2018/7/6 21:16
# @Author  : cap
# @FileName: iris_data.py
# @Software: PyCharm Community Edition
# @introduction:
import pandas as pd
import tensorflow as tf

TRAIN_URL = 'D:/softfiles/workspace/data/tensorflow/data/iris_training.csv'
TEST_URL = 'D:/softfiles/workspace/data/tensorflow/data/iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# 加载本地数据，
def load_data(y_name='Species'):
    train_path, test_path = TRAIN_URL, TEST_URL

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    # 把数据集转成dataset格式
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, 'batch_size must not None'
    dataset = dataset.batch(batch_size)
    return dataset