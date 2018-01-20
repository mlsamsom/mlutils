from ..datautils import HDF5Batcher
from layers import conv_relu_pool, fully_connected

import tensorflow as tf
import numpy as np
from datetime import datetime


def train_simple_conv_net(ckpt_dir, learning_rate, epochs, img_size,
                          num_classes, batcher, batch_size=50):
    """A very simple conv net implementation for testing the library

    Define the graph

                            input
                                |
                2d convolution (5x5, stride=1)
                                |
                            ReLU
                                |
                    Max Pool (2x2, stride=2)
                                |
                2d convolution (5x5, stride=1)
                                |
                            ReLU
                                |
                    Max Pool (2x2, stride=2)
                                |
                    Fully connected (1024)
                                |
                    Fully connected (num_classes)

    You can either define the graph inline here or pull it from
    another function.

    Please call the output logits

    Args:
    img_size : tuple
        The dimentions of the input image
    num_classes : int
        Number of classes training on
    train : bool
        Whether the graph is in training mode
    """
    print("[INFO] Initializing new graph")
    with self.graph.as_default():
        # 1. Input layer
        # -------------------------------------------------------
        input_shape = [None] + list(img_size)
        X = tf.placeholder(tf.float32, shape=img_shape)
        y_true = tf.placeholder(tf.float32, shape=num_classes)

        # 2. Convolution-ReLU-Pool layer 1
        # -------------------------------------------------------
        # 32 filters, 5 by 5 window
        conv1_params = {'shape': [5, 5, input_shape[-1], 32],
                        'strides': [1, 1, 1, 1],
                        'padding': 'SAME'}

        pool1_params = {'shape': [1, 2, 2, 1],
                        'strides': [1, 2, 2, 1],
                        'padding': 'SAME'}

        conv1 = conv_relu_pool(X, conv1_params, pool1_params)

        # -------------------------------------------------------

        # 3. Convolution-ReLU-Pool layer 2
        # -------------------------------------------------------
        conv2_params = {'shape': [5, 5, conv2_params[-1], 64],
                        'strides': [1, 1, 1, 1],
                        'padding': 'SAME'}
        pool2_params = pool1_params

        conv2 = conv_relu_pool(conv1, conv2_params, pool2_params)
        # -------------------------------------------------------

        # 4. Fully connected layer 1
        # -------------------------------------------------------
        full1 = fully_connected(conv2, 1024)
        # -------------------------------------------------------

        # 5. ReLU Activation
        # -------------------------------------------------------
        full1 = tf.nn.relu(full1)
        # -------------------------------------------------------

        # Dropout
        hold_prob = tf.placeholder(tf.float32)
        full1 = tf.nn.dropout(full1, keep_prob=hold_prob)

        # 6. Fully connected layer 2
        # -------------------------------------------------------
        logits = fully_connected(full1, num_classes)
        # -------------------------------------------------------

        # TRAINING VARIABLES
        # -------------------------------------------------------
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                    logits=logits)
        )
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        train = optimizer.minimize(cross_entropy)
        # -------------------------------------------------------
        init = tf.global_variables_initializer()

        # VALIDATION VARIABLES
        # -------------------------------------------------------
        matches = tf.equal(tf.argmax(logits, 1),
                            tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        # -------------------------------------------------------

    saver = tf.train.Saver()
    with tf.Session(graph=self.graph):
        sess.run(init)
        for i in range(self.epochs):
            X_train, y_train = batcher.next_batch('train', batch_size)
            feed_dict = {X: X_train, y_true: y_train, hold_prob: 0.5}
            sess.run(train, feed_dict=feed_dict)

            if i%100 == 0:
                save_path = saver.save(sess, "{}/int.ckpt".format(ckpt_dir))
                val_X, val_y = batcher.get_all('test')
                feed_dict = {X: val_X, y_true: val_y, hold_prob: 1.0}
                epoch_acc = sess.run(acc, feed_dict=feed_dict)
                print("Epoch: {}".format(i))
                print("Accuracy: {}".format(racc))
        save_path = saver.save(sess, "{}/final_model.ckpt".format(ckpt_dir))

    return self


if __name__ == "__main__":
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    ckpt_dir = '../../data/simple_cifar/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    batcher = HDF5Batcher('../../data/data.hdf5')

    train_simple_conv_net(ckpt_dir,
                          learning_rate=0.001,
                          epochs=100,
                          img_size=(32, 32, 3),
                          num_classes=10,
                          batcher=batcher)
