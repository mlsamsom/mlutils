import tensorflow as tf
import numpy as np

from graph import define_graph
from data import TimeSeriesData
from config import Config as cfg


ts_data = TimeSeriesData(250, 0, 10)

g = define_graph(cfg.num_time_steps, cfg.num_inputs, cfg.num_outputs,
                 cfg.num_neurons, cfg.learning_rate)

with tf.Session(graph=g) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for iteration in np.arange(cfg.num_train_iterations):
        X_batch, y_batch = ts_data.next_batch(cfg.batch_size, cfg.num_time_steps)
        sess.run(train, feed_dict={X:X_batch, y:y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print("{}\tMSE: {}".format(iteration, mse))

    saver.save(sess, "./saved_model/rnn_time_series_model")
