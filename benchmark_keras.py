import argparse
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--backend")
args = parser.parse_args()

os.environ['KERAS_BACKEND'] = args.backend

import tensorflow.python.keras.backend

if args.backend == "theano":
    tensorflow.python.keras.backend.set_image_dim_ordering("th")
elif args.backend == "tensorflow":
    tensorflow.python.keras.backend.set_image_dim_ordering("tf")
else:
    print("Backend must be theano or tensorflow")

tensorflow.python.keras.backend.set_floatx('float32')
tensorflow.python.keras.backend.set_epsilon(1e-07)

import numpy as np
import time
import sys
import tensorflow
# from tensorflow.keras.optimizers import SGD

from tensorflow.python.keras.applications.vgg16 import VGG16

width = 224
height = 224
batch_size = 16

model = VGG16(include_top=True, weights=None)
sgd = tensorflow.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd) # loss='hinge'

if args.backend == "theano":
    x = np.zeros((batch_size, 3, width, height), dtype=np.float32)
elif args.backend == "tensorflow":
    x = np.zeros((batch_size, width, height, 3), dtype=np.float32)
else:
    print("Backend must be theano or tensorflow")
y = np.zeros((batch_size, 1000), dtype=np.float32)

# warmup
model.train_on_batch(x, y)

t0 = time.time()
n = 0
while n < 100:
    tstart = time.time()
    model.train_on_batch(x, y)
    tend = time.time()
    print("Iteration: %d train on batch time: %7.3f ms." %( n, (tend - tstart)*1000 ))
    n += 1
t1 = time.time()

print("Batch size: %d" %(batch_size))
print("Iterations: %d" %(n))
print("Time per iteration: %7.3f ms" %((t1 - t0) *1000 / n))
