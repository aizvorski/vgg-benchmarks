import mxnet as mx
import numpy as np
import time


def vgg16_symbol():
    data = mx.sym.Variable('data')

    # conv1
    conv1_1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.sym.Pooling(data=relu1_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")

    # conv2
    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.sym.Pooling(data=relu2_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")

    # conv3
    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.sym.Pooling(data=relu3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")

    # conv4
    conv4_1 = mx.sym.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.sym.Pooling(data=relu4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")

    # conv5
    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.sym.Pooling(data=relu5_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool5")

    # fc6
    flat6 = mx.sym.Flatten(data=pool5, name="flat6")
    fc6 = mx.sym.FullyConnected(data=flat6, num_hidden=4096, name="fc6")
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")

    # fc7
    fc7 = mx.sym.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")

    # fc8
    fc8 = mx.sym.FullyConnected(data=drop7, num_hidden=1000, name="fc8")
    softmax = mx.sym.SoftmaxOutput(data=fc8, name="softmax")

    return softmax


batch_size = 16
nb_epoch = 400

imgs = np.random.uniform(0, 1, (batch_size * 3 * 224 * 224))
imgs = imgs.reshape((batch_size, 3, 224, 224))
labels = np.random.randint(0,1000, batch_size)

train_iter = mx.io.NDArrayIter(data=imgs.astype(np.float32),
                               label=labels.astype(np.float32),
                               batch_size=batch_size)

vgg16 = vgg16_symbol()

model = mx.model.FeedForward(
    ctx=mx.gpu(),
    symbol=vgg16,
    num_epoch=nb_epoch,
    learning_rate=0.01,
    momentum=0.9,
    wd=0.00001,
    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
)

# A warmup run did not improve the benchmark.
# As the first few epochs are significantly longer than the others,
# we run over more epochs (400) to smooth them out
t0 = time.time()
model.fit(X=train_iter)
t1 = time.time()
print "Batch size: %d" % (batch_size)
print "Iterations: %d" % (nb_epoch)
print "Time per iteration: %7.3f ms" % ((t1 - t0) * 1000 / nb_epoch)
