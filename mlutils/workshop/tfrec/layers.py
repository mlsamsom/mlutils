import tensorflow as tf


def conv_relu_pool(X, conv_params, pool_params):
    """
    Initializes weights and biases
    and does a 2d conv-relu-pool
    """
    # Initialize weights
    W = tf.Variable(
        tf.truncated_normal(conv_params, stddev=0.1)
    )
    b = tf.constant(0.1, shape=conv_params['shape'])
    conv = tf.nn.conv2d(X, W,
                        strides=conv_params['strides'],
                        padding=conv_params['padding'])

    # Simple ReLU activation function
    conv = tf.nn.relu(conv)

    #  2 by 2 max ppoling with a stride of 2
    out = tf.nn.max_pool(conv,
                            ksize=pool_params['shape'],
                            strides=pool_params['strides'],
                            padding=pool_params['padding'])
    return out


def fully_connected(X, size):
    # Reshape
    input_size = int(sum(X.get_shape()[1:]))
    X = tf.reshape(X, [-1, input_size])

    # Initialize weights
    W = tf.Variable(
        tf.truncated_normal([input_size, size], stddev=0.1)
    )
    b = tf.constant(0.1, shape=[size])
    out = tf.matmul(X, W) + b
    return out
