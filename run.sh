#!bash

if [ -d deep-learning-models ];
then
   cd deep-learning-models ; git pull ; cd ..
else
   git clone https://github.com/fchollet/deep-learning-models
fi

nvidia-smi dmon > results/benchmark_caffe.dmon 2>&1 &
python benchmark_caffe.py > results/benchmark_caffe.output 2>&1
killall -9 nvidia-smi

nvidia-smi dmon > results/benchmark_mxnet.dmon 2>&1 &
python benchmark_mxnet.py > results/benchmark_mxnet.output 2>&1
killall -9 nvidia-smi

# PATH=/usr/local/cuda/bin/:$PATH python benchmark_neon.py > results/benchmark_neon.output

nvidia-smi dmon > results/benchmark_tensorflow.dmon 2>&1 &
python benchmark_tensorflow.py --train_schedule pure_tf > results/benchmark_tensorflow.output 2>&1
killall -9 nvidia-smi

nvidia-smi dmon > results/benchmark_tensorflow_slim.dmon 2>&1 &
python benchmark_tensorflow.py --train_schedule slim > results/benchmark_tensorflow_slim.output 2>&1
killall -9 nvidia-smi

nvidia-smi dmon > results/benchmark_keras_theano.dmon 2>&1 &
python benchmark_keras.py --backend theano     > results/benchmark_keras_theano.output 2>&1
killall -9 nvidia-smi

nvidia-smi dmon > results/benchmark_keras_tensorflow.dmon 2>&1 &
python benchmark_keras.py --backend tensorflow > results/benchmark_keras_tensorflow.output 2>&1
killall -9 nvidia-smi


