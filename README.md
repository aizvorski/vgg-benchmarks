# Simple Deep Learning Benchmark (VGG16)

This mini-benchmark compares the speed of several deep learning frameworks on the VGG-16 network architecture.

*Contributions welcome!*

## Results

Here are the results on a GTX 1080 with a minibatch size of 16.

| Framework  | Time (per minibatch)  |
|:---|:---|
| Neon  | 164.527 ms  |
| Torch (1)  | 232.55 ms  |
| Caffe  | 244.445 ms  |
| Keras (Tensorflow)  | 287.693 ms  |
| Keras (Theano)  | 409.953 ms  |

(1) The Torch benchmark is from https://github.com/jcjohnson/cnn-benchmarks (it has an essentially identical setup, VGG-16, GTX 1080, CUDA 8, cuDNN 5, minibatch size 16).

(2) The time is for a complete SGD step including parameter updates, not just the forward+backward time.

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
- Neon: 1.5.4 (485033c)
- Python: 2.7.6
- Ubuntu: 14.04.3 LTS

The TensorFlow version is *very* recent, it has to be in order for it to work with CUDA 8.0.
