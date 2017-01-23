## Results GTX 1080

The mini batch size is 16

| Framework  | Time (per minibatch)  |
|:---|:---|
| Neon  | 164.527 ms  |
| Torch (1)  | 232.55 ms  |
| Caffe (2)  | 244.445 ms  |
| Keras (TensorFlow)  | 287.693 ms  |
| Keras (Theano)  | 409.953 ms  |

The environment for the results listed above is as follows:

- Hardware: GTX 1080 (EVGA GTX 1080 Founders Edition)
- OS: Ubuntu 14.04.3 LTS
- CUDA: 8.0 (cuda-repo-ubuntu1404-8-0-rc_8.0.27-1_amd64.deb)
- CuDNN: 5.0 (cudnn-8.0-linux-x64-v5.0-ga.tgz)
- Caffe: df412ac (from source)
- Keras: 1.0.7
- Theano: 0.8.2
- TensorFlow: 85f76f5 (from source)
- Neon: 1.5.4 (485033c)
- Python: 2.7.6

The TensorFlow version was *very* recent, it has to be in order for it to work with CUDA 8.0.

(1) The Torch benchmark is from https://github.com/jcjohnson/cnn-benchmarks (it has an essentially identical setup, VGG-16, GTX 1080, CUDA 8, cuDNN 5, minibatch size 16).

(2) The time is for a complete SGD step including parameter updates, not just the forward+backward time.
