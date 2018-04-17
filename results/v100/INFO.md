## Results V100

The mini batch size is 16

| Framework | Time (per minibatch)  |
|:---|:---|
| TensorFlow | 117.11 | 
| TensorFlow (slim) | 176.14 | 
| Keras (TensorFlow) | 149.85 | 
| MXNet | 81.22 | 

- Hardware: Tesla V100-SXM2-16GB (Amazon AWS EC2 p3.2xlarge)
- OS: Ubuntu 16.04.3 LTS (NVIDIA Volta Deep Learning AMI ami-b933cac4)
- Python: 2.7.6
- CUDA: 9.0.333
- CuDNN: ?
- TensorFlow: 1.4.0
- Keras: 2.0.8-tf
- MXNet: 1.1.0
- Caffe: n/a
- Theano: n/a
- Neon: n/a

(1) Tensorflow and Keras were run in NGC docker container nvcr.io/nvidia/tensorflow:18.03-py3, and MXNet in nvcr.io/nvidia/mxnet:18.03-py3
(2) The time is for a complete SGD step including parameter updates, not just the forward+backward time.
