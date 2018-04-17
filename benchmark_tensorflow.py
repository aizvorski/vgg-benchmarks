import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import argparse


def vgg16(inputs, num_classes, batch_size):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding="SAME", scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], padding="SAME", scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding="SAME", scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], padding="SAME", scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], padding="SAME", scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = tf.reshape(net, (batch_size, 7 * 7 * 512))
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net


def train_slim(batch_size, height, width, num_classes, learning_rate):
    """Built-in slim training schedule"""

    # number of iterations
    n = 100

    logdir = None  # Don't store checkpoints

    with tf.Session():
        train_inputs = tf.random_uniform((batch_size, height, width, 3))
        labels = tf.one_hot(np.arange(batch_size), on_value=1.0, off_value=0.0, depth=num_classes)

        predictions = vgg16(train_inputs, num_classes, batch_size)
        loss = slim.losses.softmax_cross_entropy(predictions, labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        train_op = slim.learning.create_train_op(loss, optimizer)

        t0 = time.time()
        slim.learning.train(
            train_op,
            logdir,
            number_of_steps=n,
            save_summaries_secs=3000,
            save_interval_secs=6000)
        t1 = time.time()

        print("Batch size: %d" % (batch_size))
        print("Iterations: %d" % (n))
        print("Time per iteration: %7.3f ms" % ((t1 - t0) * 1000 / n))


def train_pure_tf(batch_size, height, width, num_classes, learning_rate):
    """pure tensorflow training schedule, possibly with less overhead than slim"""

    # number of iterations
    n = 100

    with tf.Graph().as_default(), tf.device('/gpu:0'):

        train_inputs = tf.random_uniform((batch_size, height, width, 3))
        labels = tf.one_hot(np.arange(batch_size), on_value=1.0, off_value=0.0, depth=num_classes)

        # Predictions
        predictions = vgg16(train_inputs, num_classes, batch_size)
        # Loss function
        loss = slim.losses.softmax_cross_entropy(predictions, labels)
        # Optimizer
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        # Calculate the gradients for the batch of data
        grads = opt.compute_gradients(loss)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads)

        # Run a session
        with tf.Session() as sess:
            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()
            sess.run(init)

            # warmup run (generally the first run is much slower than the others)
            sess.run([apply_gradient_op])

            t0 = time.time()
            for i in range(n):
                tstart = time.time()
                sess.run([apply_gradient_op])
                tend = time.time()
                print("Iteration: %d train on batch time: %7.3f ms." % (i, (tend - tstart) * 1000))
            t1 = time.time()

        print("Batch size: %d" % (batch_size))
        print("Iterations: %d" % (n))
        print("Time per iteration: %7.3f ms" % ((t1 - t0) * 1000 / n))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_schedule", default="pure_tf")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    if args.train_schedule == "slim":
        train_slim(args.batch_size, args.height, args.width, args.num_classes, args.learning_rate)
    elif args.train_schedule == "pure_tf":
        train_pure_tf(args.batch_size, args.height, args.width, args.num_classes, args.learning_rate)
    else:
        print("Train schedule must be slim or pure_tf")
