import tensorflow as tf

# 获取文件列表,需要初始化
files = tf.train.match_filenames_once('./tfrecord/data-*')

# 创建输入队列
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 创建reader
reader = tf.TFRecordReader()

# 读取文件列表
_, serialized_example = reader.read(filename_queue)

# 解析每个feature
features = tf.parse_single_example(serialized_example, features={
    'i': tf.FixedLenFeature([], tf.int64),
    'j': tf.FixedLenFeature([], tf.int64)
})

with tf.Session() as sess:
    tf.local_variables_initializer().run()

    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6):
        print(i)
        print(sess.run([features['i'], features['j']]))

    coord.request_stop()
    coord.join(threads)