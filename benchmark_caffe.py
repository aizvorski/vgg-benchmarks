import subprocess
import shutil

for n in range(1,16):
    shutil.copy("images/img0000.tiff", "images/img%04d.tiff"%(n))

CAFFE = "caffe/build/tools/caffe"

# this benchmarks forward+backward rather than a full SGD step; it turns out there is quite a significant difference between the two
# subprocess.call(CAFFE + " time -model vgg16_deploy.prototxt -iterations 100 -gpu 0", shell=True)

subprocess.call(CAFFE + " train --solver vgg16_solver.prototxt -gpu 0", shell=True)

