import tensorflow as tf

# 创建一个先进先出的队列
queue = tf.FIFOQueue(10, 'float')

# 获取一个随机正太分布的数，执行进队操作。
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 创建五个线程执行入队操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coor = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coor)
    for _ in range(3):
        print(sess.run(out_tensor))

    coor.request_stop()
    coor.join(threads)
