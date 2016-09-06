"""
From https://raw.githubusercontent.com/ry/tensorflow-vgg16/master/vgg16.py
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim


# class VGG16():
#     def __init__(self, vgg16_npy_path=None):

#         self.data_dict = {} # np.load(vgg16_npy_path, encoding='latin1').item()

#     def get_conv_filter(self, name):
#         return tf.constant(self.data_dict[name][0], name="filter")

#     def get_bias(self, name):
#         return tf.constant(self.data_dict[name][1], name="biases")

#     def get_fc_weight(self, name):
#         return tf.constant(self.data_dict[name][0], name="weights")

#     def _max_pool(self, bottom, name):
#         return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#             padding='SAME', name=name)

#     def _conv_layer(self, bottom, name):
#         with tf.variable_scope(name) as scope:
#             filt = self.get_conv_filter(name)
#             conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

#             conv_biases = self.get_bias(name)
#             bias = tf.nn.bias_add(conv, conv_biases)

#             relu = tf.nn.relu(bias)
#             return relu

#     def _fc_layer(self, bottom, name):
#         with tf.variable_scope(name) as scope:
#             shape = bottom.get_shape().as_list()
#             dim = 1
#             for d in shape[1:]:
#                  dim *= d
#             x = tf.reshape(bottom, [-1, dim])

#             weights = self.get_fc_weight(name)
#             biases = self.get_bias(name)

#             # Fully connected layer. Note that the '+' operation automatically
#             # broadcasts the biases.
#             fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

#             return fc

#     # Input should be an rgb image [batch, height, width, 3]
#     # values scaled [0, 1]
#     def build(self, rgb, train=False):

#         self.relu1_1 = self._conv_layer(rgb, "conv1_1")
#         self.relu1_2 = self._conv_layer(self.relu1_1, "conv1_2")
#         self.pool1 = self._max_pool(self.relu1_2, 'pool1')

#         self.relu2_1 = self._conv_layer(self.pool1, "conv2_1")
#         self.relu2_2 = self._conv_layer(self.relu2_1, "conv2_2")
#         self.pool2 = self._max_pool(self.relu2_2, 'pool2')

#         self.relu3_1 = self._conv_layer(self.pool2, "conv3_1")
#         self.relu3_2 = self._conv_layer(self.relu3_1, "conv3_2")
#         self.relu3_3 = self._conv_layer(self.relu3_2, "conv3_3")
#         self.pool3 = self._max_pool(self.relu3_3, 'pool3')

#         self.relu4_1 = self._conv_layer(self.pool3, "conv4_1")
#         self.relu4_2 = self._conv_layer(self.relu4_1, "conv4_2")
#         self.relu4_3 = self._conv_layer(self.relu4_2, "conv4_3")
#         self.pool4 = self._max_pool(self.relu4_3, 'pool4')

#         self.relu5_1 = self._conv_layer(self.pool4, "conv5_1")
#         self.relu5_2 = self._conv_layer(self.relu5_1, "conv5_2")
#         self.relu5_3 = self._conv_layer(self.relu5_2, "conv5_3")
#         self.pool5 = self._max_pool(self.relu5_3, 'pool5')

#         self.fc6 = self._fc_layer(self.pool5, "fc6")
#         assert self.fc6.get_shape().as_list()[1:] == [4096]

#         self.relu6 = tf.nn.relu(self.fc6)
#         if train:
#             self.relu6 = tf.nn.dropout(self.relu6, 0.5)

#         self.fc7 = self._fc_layer(self.relu6, "fc7")
#         self.relu7 = tf.nn.relu(self.fc7)
#         if train:
#             self.relu7 = tf.nn.dropout(self.relu7, 0.5)

#         self.fc8 = self._fc_layer(self.relu7, "fc8")
#         self.prob = tf.nn.softmax(self.fc8, name="prob")

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CHANNELS = 3

train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

# model = VGG16().build(train_data_node, train=True)

def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net


train_batch_size = 32
eval_batch_size = 32
train_height, train_width = 224, 224
eval_height, eval_width = 224, 224
num_classes = 1000
with tf.Session() as sess:
    train_inputs = tf.random_uniform( (train_batch_size, train_height, train_width, 3))
    net = vgg16(train_inputs)
    # tf.get_variable_scope().reuse_variables()
    # eval_inputs = tf.random_uniform(
    #   (eval_batch_size, eval_height, eval_width, 3))
    # logits, _ = vgg.vgg_16(eval_inputs, is_training=False, spatial_squeeze=False)
    # logits = tf.reduce_mean(logits, [1, 2])
    # predictions = tf.argmax(logits, 1)
