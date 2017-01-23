# Simple Deep Learning Benchmark (VGG16)

This mini-benchmark compares the speed of several deep learning frameworks on the VGG-16 network architecture.

*Contributions welcome!*

## Results

All times in milliseconds per minibatch on a VGG16 network.  Minibatch size is 16. The time is for a complete SGD step including parameter updates, not just the forward+backward time.

| Framework | [GTX 1080](results/gtx_1080/INFO.md) | [Maxwell Titan X](results/maxwell_titan_x/INFO.md) | [K80](results/k80/INFO.md) | [K520](results/k520/INFO.md) | 
| Neon | 164.53 | 207.41 | N/A | N/A | 
| Caffe | 244.44 | 311.06 | 787.81 | OOM | 
| Keras (TensorFlow) | 287.69 | 360.75 | 1021.81 | OOM | 
| Keras (Theano) | 409.95 | 317.30 | 1141.79 | 2445.22 | 
| TensorFlow | N/A | 332.27 | 1057.12 | 2290.51 | 
| TensorFlow (slim) | N/A | 370.89 | 1126.70 | 2488.51 | 
| MXNet | N/A | 324.63 | 1247.47 | OOM | 
| Torch (1) | 232.55 | 273.54 | N/A | N/A |

N/A - test not ran
OOM - test ran but failed due to running out of memory (several tests on the K520 with only 4GB memory)

(1) The Torch benchmark is from https://github.com/jcjohnson/cnn-benchmarks (it is not included in this repo).


## Running


```
bash run.sh
```

Note: this will back up and then restore your ~/.keras/keras.json

Caffe should be built inside caffe/ in the current directory (or a symlink).

Neon should be built anywhere and (just for the Neon test) the built Neon virtualenv should be activated.


