import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def full_connection():
    """
    用全连接对手写数字进行识别
    :return:
    """
    # 1）准备数据
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    # 用占位符定义真实数据
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 2）构造模型 - 全连接
    # [None, 784] * W[784, 10] + Bias = [None, 10]
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], stddev=0.01))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10], stddev=0.1))
    y_predict = tf.matmul(X, weights) + bias

    # 3）构造损失函数
    loss_list = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_true)
    loss = tf.reduce_mean(loss_list)

    # 4）优化损失
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    # 5）增加准确率计算
    bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init)

        # 开始训练
        for i in range(5000):
            # 获取真实值
            image, label = mnist.train.next_batch(500)

            _, loss_value, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={X: image, y_true: label})

            print("第%d次的损失为%f，准确率为%f" % (i+1, loss_value, accuracy_value))


    return None

if __name__ == "__main__":
    full_connection()