import subprocess

CAFFE = "caffe/build/tools/caffe"
subprocess.call(CAFFE + " time -model vgg16_deploy.prototxt -iterations 100 -gpu 0", shell=True)
