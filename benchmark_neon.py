#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
VGG_A Benchmark
https://github.com/soumith/convnet-benchmarks

./vgg_a.py
./vgg_a.py -d f16
"""

from neon import NervanaObject
from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, GlorotUniform, Xavier
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti
from neon.models import Model
from neon.data import ArrayIterator
from neon.callbacks.callbacks import Callbacks

import numpy as np
parser = NeonArgparser(__doc__)
args = parser.parse_args()

NervanaObject.be.bsz = 16
NervanaObject.be.enable_winograd = 4

# setup data provider
X_train = np.random.uniform(-1, 1, (16, 3 * 224 * 224))
y_train = np.random.randint(0, 999, (16, 1000))
train = ArrayIterator(X_train, y_train, nclass=1000, lshape=(3, 224, 224))

# layers = [Conv((3, 3, 64), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Pooling(2, strides=2),
#           Conv((3, 3, 128), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Pooling(2, strides=2),
#           Conv((3, 3, 256), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Conv((3, 3, 256), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Pooling(2, strides=2),
#           Conv((3, 3, 512), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Conv((3, 3, 512), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Pooling(2, strides=2),
#           Conv((3, 3, 512), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Conv((3, 3, 512), init=Gaussian(scale=0.01), activation=Rectlin(), padding=1),
#           Pooling(2, strides=2),
#           Affine(nout=4096, init=Gaussian(scale=0.01), activation=Rectlin()),
#           Affine(nout=4096, init=Gaussian(scale=0.01), activation=Rectlin()),
#           Affine(nout=1000, init=Gaussian(scale=0.01), activation=Softmax())]
# model = Model(layers=layers)

"""
Modified from https://github.com/NervanaSystems/ModelZoo/blob/master/ImageClassification/ILSVRC2012/VGG/vgg_neon.py
"""

init1 = Xavier(local=True)
initfc = GlorotUniform()

relu = Rectlin()
conv_params = {'init': init1,
               'strides': 1,
               'padding': 1,
               'bias': Constant(0),
               'activation': relu}

# Set up the model layers
layers = []
for nofm in [64, 128, 256, 512, 512]:
    layers.append(Conv((3, 3, nofm), **conv_params))
    layers.append(Conv((3, 3, nofm), **conv_params))
    if nofm > 128:
        layers.append(Conv((3, 3, nofm), **conv_params))
        # if args.vgg_version == 'E':
        #     layers.append(Conv((3, 3, nofm), **conv_params))
    layers.append(Pooling(2, strides=2))

layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=1000, init=initfc, bias=Constant(0), activation=Softmax()))

model = Model(layers=layers)

weight_sched = Schedule([22, 44, 65], (1 / 250.)**(1 / 3.))
opt_gdm = GradientDescentMomentum(0.01, 0.0, wdecay=0.0005, schedule=weight_sched)
opt = MultiOptimizer({'default': opt_gdm})
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
callbacks = Callbacks(model)

import time

num_epochs=100
t0 = time.time()
# model.benchmark(train, cost=cost, optimizer=opt, niterations=100, nskip=1)
model.fit( train, cost=cost, optimizer=opt, num_epochs=100, callbacks=callbacks )
t1 = time.time()

print "Batch size: %d" %(NervanaObject.be.bsz)
print "Iterations: %d" %(num_epochs)
print "Time per iteration: %7.3f ms" %((t1 - t0) *1000 / num_epochs)

