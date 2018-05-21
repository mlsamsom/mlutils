import tensorflow as tf
import numpy as np
import os
import copy
import pickle

from sklearn.preprocessing import MinMaxScaler


def next_batch(training_data, batch_size, steps):
    rand_start = np.random.randint(0, training_data.size - steps)
    y_batch = np.array(training_data[:, rand_start: rand_start+steps+1])
    return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


class LSTMPredictor(object):
    """Very simple RNN to predict a time series functions
    """
    def __init__(self, config):
        """Instantiate deep learning model

        Args:
            config (dict): model configuration file
        """
        # unpack config
        self.config = copy.deepcopy(config)
        self.result_dir = config['result_dir']
        self.num_time_steps = config['num_time_steps']
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']
        self.num_neurons = config['num_neurons']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.scaler = None
        device = config['device']
        if 'cpu' in device.lower() or 'gpu' in device.lower():
            self.device = device
        else:
            self.device = None

        self.lr = config['learning_rate']

        self.graph = tf.Graph()
        self.build_graph()

        # instantiate saver
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)

        self._init()

    def build_graph(self):
        """Define your graph here

        Abstract your graph with functions as needed
        """
        with self.graph.as_default():
            with tf.device(self.device):
                with tf.variable_scope('placeholder'):
                    self.X = tf.placeholder(
                        tf.float32,
                        [None, self.num_time_steps, self.num_inputs],
                        name='inputs'
                    )
                    self.y = tf.placeholder(
                        tf.float32,
                        [None, self.num_time_steps, self.num_outputs],
                        name='labels'
                    )
                with tf.variable_scope('encoding'):
                    self.gcell = tf.contrib.rnn.BasicLSTMCell(
                        num_units=self.num_neurons, activation=tf.nn.relu
                    )
                    self.wcell = tf.contrib.rnn.OutputProjectionWrapper(
                        self.gcell, output_size=self.num_outputs
                    )
                    self.outputs, self.states = tf.nn.dynamic_rnn(
                        self.wcell, self.X, dtype=tf.float32
                    )
                with tf.variable_scope('loss'):
                    self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))

                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.train = optimizer.minimize(self.loss)

    def infer(self, X_seed, iters):
        """Run forward inference on graph

        Args:
            X_seed (ndarray): Define size of input here
            iters (int): Time in the future to predict to
        """
        X_seed = self.scaler.transform(X_seed.T).T
        X_seed = list(np.squeeze( X_seed[:, -12:] ))

        for iteration in np.arange(iters):
            X_batch = np.array(X_seed[-self.num_time_steps:]).reshape(
                1, self.num_time_steps, 1)
            y_pred = self.sess.run(self.outputs, feed_dict={self.X: X_batch})
            X_seed.append(y_pred[0, -1, 0])

        results = self.scaler.inverse_transform(np.array(X_seed[12:]).reshape(12,1))
        return np.squeeze(results)

    def learn_from_epoch(self, X_batch, y_batch):
        """Function for learning from single epoch
        """
        self.sess.run(self.train, feed_dict={self.X: X_batch, self.y: y_batch})

    def fit(self, X_train, save_name='lstm_predictor'):
        """Train the model
        """
        self.sess.run(self.init)

        X_train = self.scaler.fit_transform(X_train.T).T

        for epoch in np.arange(self.epochs):
            X_batch, y_batch = next_batch(X_train, self.batch_size, self.num_time_steps)
            self.learn_from_epoch(X_batch, y_batch)

            if epoch % 100 == 0:
                mse = self.sess.run(self.loss, feed_dict={self.X: X_batch, self.y: y_batch})
                print("Epoch {} MSE: {}".format(epoch, mse))
        save_path = os.path.join(self.result_dir, save_name+"_final")
        self.save_model(save_path)

    def save_model(self, save_path):
        """Save the model
        """
        self.saver.save(self.sess, save_path)
        pickle.dump(self.scaler, open(self.result_dir+'/scaler.p', 'wb'))

    def _init(self):
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is not None:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.scaler = pickle.load(open(self.result_dir+'/scaler.p', 'rb'))
            print("Loaded model")
        else:
            self.scaler = MinMaxScaler()

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
