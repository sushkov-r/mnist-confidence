import tensorflow as tf
import numpy as np


def to_onehot(array, depth):
    sparse = np.zeros((len(array), depth))
    sparse[np.arange(len(array)), array] = 1
    return sparse


def make_mnist_data():
    TRAIN_SIZE = 50000
    (x_tr_val, y_tr_val), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_tr, x_val = np.split(x_tr_val, [TRAIN_SIZE])
    y_tr, y_val = np.split(y_tr_val, [TRAIN_SIZE])

    y_tr, y_val, y_test = [to_onehot(y, 10) for y in [y_tr, y_val, y_test]]
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


def data_generator(arrays, batch_size):
    idxs = list(range(len(arrays[0])))
    while True:
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            yield [array[batch_idxs] for array in arrays]


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


class Model:
    def __init__(self,
                 learning_rate=0.001,
                 num_steps=500,
                 batch_size=128,
                 predict_confidence=True
                 ):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.predict_confidence = predict_confidence

        self.display_step = 10
        self.im_width = 28
        self.num_classes = 10
        self.dropout = 0.75

        (self.x_tr, self.y_tr), (self.x_val, self.y_val), (self.x_test, self.y_test) = make_mnist_data()

        # model input (MNIST image) and target label
        self.X = tf.placeholder(tf.float32, [None, self.im_width, self.im_width])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # prediction of the model
        cnn_pred = self.conv_net(self.X, self.keep_prob)

        logits = cnn_pred[:, 0:10]
        self.prediction = tf.nn.softmax(logits)

        std_dev = cnn_pred[:, -1] / 100
        self.std_dev = tf.exp(std_dev)

        if self.predict_confidence:
            self.loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.Y)
                / (2*self.std_dev+0.0005) + 0.5*tf.log(self.std_dev+0.0005)
            )
        else:
            self.loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.Y)
            )

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def conv_net(self, x, dropout):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(x, 16, [3, 3])
        conv1 = maxpool2d(conv1, k=2)

        conv2 = tf.layers.conv2d(conv1, 32, [3, 3])
        conv2 = maxpool2d(conv2, k=2)

        conv3 = tf.layers.conv2d(conv2, 32, [3, 3])
        conv3 = maxpool2d(conv2, k=2)

        fc1 = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.layers.dense(fc1, self.num_classes+1)
        return out

    def train(self):
        tr_gen = data_generator([self.x_tr, self.y_tr], 128)
        for step in range(1, self.num_steps+1):
            batch_x, batch_y = next(tr_gen)
            self.sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: self.dropout})
            if step % self.display_step == 0 or step == 1:
                loss, acc = self.sess.run([self.loss_op, self.accuracy],
                                        feed_dict={self.X: batch_x, self.Y: batch_y,
                                                self.keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " +
                    "{:.4f}".format(loss) + ", Training Accuracy= " +
                    "{:.3f}".format(acc))

    def test_random(self):
        test_gen = data_generator([self.x_test, self.y_test], 512)
        test_batch_x, test_batch_y = next(test_gen)
        print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_batch_x,
                                                                           self.Y: test_batch_y,
                                                                           self.keep_prob: 1.0}))

    def predict_test(self, size):
        test_gen = data_generator([self.x_test, self.y_test], size)
        test_batch_x, test_batch_y = next(test_gen)
        prediction, correct, std_dev = self.sess.run([self.prediction, self.correct_pred, self.std_dev],
                                      feed_dict={self.X: test_batch_x, self.Y: test_batch_y, self.keep_prob: 1.0})
        return prediction, correct, std_dev


