import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import yaml

from data import TimeSeriesData
from models.graph import SimpleRNN


CONFIGFILE = "./config/test_configuration.yaml"

with open(CONFIGFILE, "r") as f:
    config = yaml.load(f)


ap = ArgumentParser()
ap.add_argument('--inspect_data', action='store_true', default=False,
                help="plot training data for inspection")
ap.add_argument('--train', action='store_true', default=False,
                help="Run training")
ap.add_argument('--test', action='store_true', default=False,
                help="Run test")
args = ap.parse_args()

ts_data = TimeSeriesData(250, 0, 10)
model = SimpleRNN(config)
train_inst = np.linspace(
    5,
    5 + ts_data.resolution*(config['num_time_steps']),
    config['num_time_steps'] + 1
)

if args.inspect_data:
    plt.title("A training instance")
    X_train = train_inst[:-1]
    y_train = ts_data.ret_true(X_train)
    X_label = train_inst[1:]
    y_label = ts_data.ret_true(X_label)
    plt.plot(X_train, y_train, 'bo', markersize=15, alpha=0.5, label="Instance")
    plt.plot(X_label, y_label, 'ko', markersize=7, label='Target')
    plt.legend()
    plt.show()
elif args.train:
    model.fit(ts_data)
    model.close()
elif args.test:
    X_new = np.sin(
        np.array(
            train_inst[:-1].reshape(-1, config['num_time_steps'], config['num_inputs'])
        )
    )
    y_pred = model.infer(X_new)
    plt.title("Testing model")
    plt.plot(train_inst[:-1],
             np.sin(train_inst[:-1]),
             'bo',
             markersize=15,
             alpha=0.5,
             label="training inst")
    plt.plot(train_inst[1:],
             np.sin(train_inst[1:]),
             'ko',
             markersize=10,
             label='target')
    plt.plot(train_inst[1:],
             y_pred[0, :, 0],
             'r.',
             markersize=10,
             label='predictions')
    plt.xlabel('time')
    plt.legend()
    plt.show()
else:
    print("USAGE:")
    print(" To plot an example of training data")
    print(" $ python main.py --inspect_data")
    print(" To run training")
    print(" $ python main.py --train")
    print(" To test the model")
    print(" $ python main.py --test")

# g = define_graph(cfg.num_time_steps, cfg.num_inputs, cfg.num_outputs,
#                  cfg.num_neurons, cfg.learning_rate)
#
# with tf.Session(graph=g) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     for iteration in np.arange(cfg.num_train_iterations):
#         X_batch, y_batch = ts_data.next_batch(cfg.batch_size, cfg.num_time_steps)
#         sess.run(train, feed_dict={X:X_batch, y:y_batch})
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
#             print("{}\tMSE: {}".format(iteration, mse))
#
#     saver.save(sess, "./saved_model/rnn_time_series_model")
