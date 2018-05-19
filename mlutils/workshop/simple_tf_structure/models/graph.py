import tensorflow as tf
import os, copy


class SimpleRNN(object):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.result_dir = config['result_dir']
        self.num_time_steps = config['num_time_steps']
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']
        self.num_neurons = config['num_neurons']
        self.lr = config['learning_rate']

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def build_graph(self):
        with self.graph.as_default():
            with tf.variable_scope('placeholder'):
                self.X = tf.placeholder(
                    tf.float32, [None, self.num_time_steps, self.num_inputs], name='inputs'
                )
                self.y = tf.placeholder(
                    tf.float32, [None, self.num_time_steps, self.num_outputs], name='labels'
                )
            with tf.variable_scope('encoding'):
                self.gcell = tf.contrib.rnn.GRUCell(
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

            self.saver = tf.train.Saver()

    def infer(self):
        pass

    def learn_from_epoch(self):
        pass

    def train(self):
        pass
