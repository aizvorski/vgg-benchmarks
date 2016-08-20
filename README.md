# Benchmark: Caffe vs Keras

This mini-benchmark compares the speed of Caffe vs Keras (using Theano) and Keras (using Tensorflow). 

It uses the VGG-16 network architecture.

## Results

Here are the results on a GTX 1080 with a minibatch size of 16.

| Framework  | Time (forward+backward)  |
|:---|:---|
| Caffe  | 244.445 ms  |
| Torch (1)  | 232.55 ms  |
| Keras (Tensorflow)  | 287.693 ms  |
| Keras (Theano)  | 409.953 ms  |

(1) The Torch benchmark is from https://github.com/jcjohnson/cnn-benchmarks (it has an essentially identical setup, VGG16, GTX 1080, CUDA 8, cuDNN 5, minibatch size 16).

## Running

Caffe should be built inside caffe/ in the current directory (or a symlink)

```bash
run.sh
```

Note: this will back up and then restore your ~/.keras/keras.json

## Environment

The environment for the results listed above is as follows:

- Hardware: GTX 1080 (EVGA GTX 1080 Founders Edition)
- CUDA: 8.0 (cuda-repo-ubuntu1404-8-0-rc_8.0.27-1_amd64.deb)
- CuDNN: 5.0 (cudnn-8.0-linux-x64-v5.0-ga.tgz)
- Caffe: df412ac (from source)
- Keras: 1.0.7
- Theano: 0.8.2
- TensorFlow: 85f76f5 (from source)
- Python: 2.7.6
- Ubuntu: 14.04.3 LTS

The TensorFlow version is *very* recent, it has to be in order for it to work with CUDA 8.0.
