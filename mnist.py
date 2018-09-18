import tensorflow as tf
import numpy as np

"""
Model for classification of MNIST digits, capable of producing prediction confidence.
See class Model for more detail.
"""

def to_onehot(array, depth):
    """
    Transform an array of labels to its one hot representation.
    """
    sparse = np.zeros((len(array), depth))
    sparse[np.arange(len(array)), array] = 1
    return sparse


def make_mnist_data():
    """
    Read the MNIST dataset and split it into training, validation and test subsets.
    """
    TRAIN_SIZE = 50000
    (x_tr_val, y_tr_val), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_tr, x_val = np.split(x_tr_val, [TRAIN_SIZE])
    y_tr, y_val = np.split(y_tr_val, [TRAIN_SIZE])

    y_tr, y_val, y_test = [to_onehot(y, 10) for y in [y_tr, y_val, y_test]]
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


def data_generator(arrays, batch_size):
    """
    A finite generator of data with shuffling.
    Arguments:
        arrays: a list of arrays containing input and target data, from which the batches are generated.
        batch_size: batch size. If the input arrays cannot be split equally, the last batch is partial.
    """
    idxs = list(range(len(arrays[0])))
    np.random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        batch_idxs = idxs[i:i+batch_size]
        yield [array[batch_idxs] for array in arrays]


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


class Model:
    """
    A simple CNN which is capable of estimating prediction confidence.
    """
    def __init__(self,
                 learning_rate=0.0003,
                 num_epochs=1,
                 batch_size=128,
                 predict_confidence=True
                 ):
        """
        Training options:
            learning_rate
            num_epochs
            batch_size
        Model option:
            predict_confidence: Boolean. Specify whether confidence prediction is required.
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.predict_confidence = predict_confidence

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

        out = tf.layers.dense(fc1, self.num_classes+1)
        return out

    def train(self):
        print("Optimizing model...")
        for epoch in range(self.num_epochs):
            tr_gen = data_generator([self.x_tr, self.y_tr], 128)
            for batch_x, batch_y in tr_gen:
                self.sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: self.dropout})

            print("Epoch {}/{}:".format(epoch+1, self.num_epochs))
            val_gen = data_generator([self.x_val, self.y_val], 10000)
            batch_x, batch_y = next(val_gen)
            loss, acc = self.sess.run([self.loss_op, self.accuracy],
                                      feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.0})

            print("Validation loss = {:.4f}, Validation accuracy = {:.3f}".format(loss, acc))

    def test_accuracy(self, num_samples=10000):
        test_gen = data_generator([self.x_test, self.y_test], 512)
        test_batch_x, test_batch_y = next(test_gen)
        accuracy = self.sess.run(self.accuracy, feed_dict={self.X: test_batch_x,
                                                           self.Y: test_batch_y,
                                                           self.keep_prob: 1.0})
        print("Testing Accuracy: ", accuracy)

    def predict_test(self, num_samples=10000):
        test_gen = data_generator([self.x_test, self.y_test], num_samples)
        test_batch_x, test_batch_y = next(test_gen)
        prediction, correct, std_dev = self.sess.run([self.prediction, self.correct_pred, self.std_dev],
                                                     feed_dict={self.X: test_batch_x,
                                                                self.Y: test_batch_y,
                                                                self.keep_prob: 1.0})
        return prediction, correct, std_dev

