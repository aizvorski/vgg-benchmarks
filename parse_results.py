from datetime import datetime
import numpy as np


def parse_output(output_file):
    """ Utility to parse the output of keras, neon and tensorflow"""

    with open("./results/%s" % output_file, "r") as f:
        list_lines = f.readlines()
        for line in list_lines:
            if "Time per iteration" in line:
                result = line.split(" ")[-2]
                return float(result)


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


def name_to_path(name):
    return name.lower().replace('(', '').replace(')', '').replace(' ', '_')


if __name__ == '__main__':
    benchmark_names = ('Neon', 'Caffe', 'Keras (TensorFlow)', 'Keras (Theano)',
                       'Tensorflow', 'Tensorflow (slim)')
    run_names = ('GTX 1080', 'Maxwell Titan X')

    line = "| Framework | "
    for run_name in run_names:
        line += run_name + " | "
    print line

    for name in benchmark_names:
        line = "| " + name + " | "
        for run_name in run_names:
            result_path = name_to_path(run_name) + "/benchmark_" + name_to_path(name) + ".output"
            if name == 'Caffe':
                time_ = parse_output_caffe(result_path)
            else:
                time_ = parse_output(result_path)
            line += "%.2f" % (time_)
            line += " | "
        print line
