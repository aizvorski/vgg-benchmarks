## Results Maxwell Titan X

The mini batch size is 16. As the speed changed with GPU heat, the fan speed was set to 100 % and the tests were all started with GPU temperature = 45 degrees.

| Framework  | Time (per minibatch)  |
|:---|:---|
| Neon  | 207.406 ms  |
| Torch (1) | 273.542 ms  |
| Caffe (2)  | 311.061 ms |
| Keras (TensorFlow)  | 360.753 ms  |
| Keras (Theano)  | 317.298 ms  |
| TensorFlow  | 332.27 ms  |
| TensorFlow (slim) (3) | 370.89 ms  |
| mxnet | 324.635 ms |

(1) The code in https://github.com/jcjohnson/cnn-benchmarks was re-run with 100 iterations, 100 % GPU fan and starting temperature of 45.

(2) The time is for a complete SGD step including parameter updates, not just the forward+backward time.

(3) Uses the built-in slim training function, which has possibly more overhead.

- Hardware: Titan X Maxwell
- OS: Ubuntu 14.04.3 LTS
- CUDA: 8.0 (cuda-repo-ubuntu1404-8-0-rc_8.0.27-1_amd64.deb)
- CuDNN: 5.0 (cudnn-8.0-linux-x64-v5.0-ga.tgz)
- Caffe: b2982c7 (from source)
- Keras: 1.0.8
- Theano: 0.9.0dev2.dev-338384adeabd2a56ccae22a9f1105a9f82ce9b8f
- TensorFlow:  2a6d751 (from source)
- Neon: 1.5.4 (485033c)
- mxnet: (066373a) from source
- Python: 2.7.6
