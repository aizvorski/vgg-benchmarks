from datetime import datetime
import numpy as np


def parse_output(output_file):
    """ Utility to parse the output of keras and neon"""

    with open("./results/%s" % output_file, "r") as f:
        list_lines = f.readlines()
        for line in list_lines:
            if "Time per iteration" in line:
                result = line.split(" ")[-2]
                return result


def parse_output_caffe(output_file):
    """ Utility to parse the output of caffe

    Count the time as the clock difference between two consecutive appearances of solver.cpp:228

    """

    # string formatting
    FMT = '%H:%M:%S.%f'

    list_time = []

    with open("./results/%s" % output_file, "r") as f:
        list_lines = f.readlines()
        for line in list_lines:
            if "solver.cpp:228" in line:
                time = line.split(" ")[1]
                time = datetime.strptime(time, FMT)
                list_time.append(time)

    arr_time = np.array(list_time)
    arr_delta = arr_time[1:] - arr_time[:-1]
    arr_delta = [d.microseconds for d in arr_delta]

    return 1E-3 * np.mean(arr_delta)

if __name__ == '__main__':

    print parse_output_caffe("benchmark_caffe.output")
    print parse_output("benchmark_keras_theano.output")
    print parse_output("benchmark_keras_tensorflow.output")
    print parse_output("benchmark_neon.output")
