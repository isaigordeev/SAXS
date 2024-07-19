import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


def get_filenames(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            if (filename != '.DS_Store'):
                yield filename


def get_filenames_without_ext(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            name, extension = os.path.splitext(filename)
            if (name != '.DS_Store'):
                yield name, extension


def load_generated_data(phase):
    data = np.load(f'../saxs_generated_data/P_{phase.lower()}.npy')
    print(data.shape)
    data_3d = data[:, :, :, 0]
    data_1d = []
    for i in data_3d:
        # get matrix diagonal
        data = np.sqrt(np.diag(i))
        data_1d.append(data)
    data_1d = np.array(data_1d)
    q = np.load(f'../saxs_generated_data/{phase.lower()}_q.npy')
    # load experimental test_processing_data
    # exp_data = np.load(f'Experimental_data/{phase.lower()}.npy')
    exp_data = None
    return data_1d, data_3d, exp_data, q


def write_generated_data_to_csv(q, I, filename):
    csv_file = filename

    data = np.array([q, I], dtype=object)
    data = np.transpose(data)
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerows(data)

    print(f'Data written to {csv_file} successfully.')


def read_data(data_dir_file):
    # print(data_dir_file)
    data_name_dir, extension = os.path.splitext(data_dir_file)

    if extension == '.csv':
        data = pd.read_csv(data_dir_file, sep=',')
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()
        if data.shape[1] >= 3:
            q = np.array(data.iloc[:, 0])
            I = np.array(data.iloc[:, 1])
            dI = np.array(data.iloc[:, 2])
            return q, I, dI
        elif data.shape[1] == 2:
            q = np.array(data.iloc[:, 0])
            I = np.array(data.iloc[:, 1])
            return q, I, None
        else:
            raise AttributeError("Input is insufficient")
    elif extension == '.npy':  # synthetic test_processing_data
        assert "processed" in data_dir_file
        data = np.load(data_dir_file)
        data_3d = data[:, :, :, 0]
        I = []
        for i in data_3d:
            # get matrix diagonal
            data = np.sqrt(np.diag(i))
            I.append(data)
        I = np.array(I)

        phase = 'cubic'
        q = np.linspace(0.001, 0.2, 500)

        exp_data = None
        return q, I[0], None


def read_I(data_dir_file, EXTENSION):
    data = pd.read_csv(data_dir_file + EXTENSION, sep=',')
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    I = np.array(data.iloc[:, 0])
    return I


def background_plot(q, I):
    plt.clf()
    plt.plot(q, I, 'bo', linewidth=0.5, label='raw_data')
    plt.show()
    # plt.savefig('1')


def calculate_absolute_difference(sequence, target):
    sequence = sequence[:len(target)]
    absolute_difference = np.sum([1 / abs(x - y) for x, y in zip(sequence, target)])
    return absolute_difference


def calculate_first_peaks(sequence, target):
    if len(target) > 0:
        sequence = sequence[:len(target)]
        absolute_difference = abs(sequence[0] - target[0])
        return 1 / absolute_difference
    else:
        print("Can not be classified")
        return None

# I, _, __, q = load_generated_data('cubic')
#
# write_generated_data_to_csv(q, I[7], 'sample.csv')
