# Benchmark: Caffe vs Keras

This mini-benchmark compares the speed of Caffe vs Keras (using Theano) and Keras (using Tensorflow). 

It uses the VGG-16 network architecture.

## Results

Here are the results on a GTX 1080 with a minibatch size of 16.

| Framework  | Time (forward+backward)  |
|:---|:---|
| Caffe  | 151.733 ms  |
| Keras (Theano)  | 539.042 ms  |
| Keras (Tensorflow)  | 776.792 ms  |

For comparison, https://github.com/jcjohnson/cnn-benchmarks lists VGG-16 on the GTX 1080 as taking 232.55 ms (Torch, cuDNN 5, minibatch size 16). The Caffe timing is surprisingly fast compared to Torch.

## Running

```bash
python benchmark_caffe.py > results/benchmark_caffe.output
python benchmark_keras.py > results/benchmark_keras.output
KERAS_BACKEND=tensorflow python benchmark_keras.py > results/benchmark_keras.tensorflow.output
```

## Environment

The environment for the results listed above is as follows:

- Hardware: GTX 1080 (Founders Edition)
- CUDA: 8.0 (cuda-repo-ubuntu1404-8-0-rc_8.0.27-1_amd64.deb)
- CuDNN: 5.0 (cudnn-8.0-linux-x64-v5.0-ga.tgz)
- Caffe: df412ac (from source)
- Keras: 1.0.7
- Theano: 0.8.2
- TensorFlow: 85f76f5 (from source)
- Python: 2.7.6
- Ubuntu: 14.04.3 LTS

The TensorFlow version is *very* recent, it has to be in order for it to work with CUDA 8.0.
