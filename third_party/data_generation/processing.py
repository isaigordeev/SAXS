import numpy as np
import random
import sys
import os

from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.special import wofz
from concurrent.futures import ProcessPoolExecutor, as_completed

# from saxs.saxs_model.tools import standartization
# from saxs.saxs_model.model_settings import IMAGE_DIM

IMAGE_DIM = 256


def standartization(sample):
    mean = np.mean(sample)
    var = np.std(sample)
    sample -= mean
    sample /= (var ** 0.5)
    sample /= np.max(sample)

    return sample



# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


# Norm functions
def norm_log(data):
    data = np.log(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def log(data):
    return np.log(data)


def minmax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Generate 2Dx3 Product matrix
def gen_product_matrix(I_values):
    return np.repeat(np.expand_dims(np.outer(I_values, I_values), -1), 3, axis=-1)


def voigtSignal(x, alpha, gamma, shift):
    """
    Return the Voigt line shape at x with x component HWHM gamma
    and Gaussian component HWHM alpha.
    Voigt profile is the convolution of the Gaussian Profile and the Lorentzian Profile
    Gamma is HWHM of Lorentzian
    Sigma is SD of Gaussian
    Alpha is HWHM of Gaussian
    https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/

    """

    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x - shift + 1j * gamma) / sigma / np.sqrt(2))) / sigma
    # return np.real(wofz((x - shift + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


def gaussianSignal(x, sigma, shift):
    """
    Return Gaussian signal at of array x with standard deviation sigma that has been shifted by shift
    """
    return np.exp((-1 / sigma) * ((x - shift)) ** 2)


def debyeWaller(q, decay_parameter, shift=0):
    """
    Return debeye waller factor which is just a gaussian decay of parameter decay_parameter
    Wikipedia - describe the attenuation of x-ray scattering or coherent neutron scattering caused by thermal motion.
    """
    return gaussianSignal(q, decay_parameter, shift)


dwf = debyeWaller


blanks = np.load(os.path.join(os.path.dirname(__file__), 'blanks_raw.npy'))


# Function for generating DWF / Voigts / Lorenztians and Interpolating test_processing_data in q2 range

def Process_cubic(data):
    random_blank = random.randint(0, 169)
    blanky = blanks[random_blank]
    q = np.linspace(0, 0.452, 1566)
    q2 = np.linspace(0.001, 0.2, 500)  # CHANGE DIM
    random_voigt_gaussalpha = random.uniform(0.0001, 0.002)
    random_voigt_lorenzgamma = random.uniform(0.0001, 0.003)
    # dw_param = random.uniform(0.1, 0.2)
    dw_param = random.uniform(0.05, 0.2)
    zero_array = np.zeros(1566)
    for i in range(len(data[0, :])):
        if not data[0, i] == 0:
            temp = random.randint(100, 300)*data[1, i] * dwf(data[0, i], dw_param) * voigtSignal(q, random_voigt_gaussalpha,
                                                                        random_voigt_lorenzgamma, data[0, i])
            zero_array += temp

            # temp = test_processing_data[1, i] * dwf(q, 1, test_processing_data[0, i])
            # zero_array += temp
        else:
            temp = data[1, i] * voigtSignal(q, random_voigt_gaussalpha, random_voigt_lorenzgamma, data[0, i])
            zero_array += temp

    # zero_array = log(zero_array+1)
    # zero_array+=1

    # I = minmax(zero_array + np.exp(-blanky))
    # I = minmax(zero_array + log(blanky))
    # I = minmax(zero_array + 1/blanky)
    I = minmax(zero_array + (blanky))
    print(max(I))
    # plt.plot(I)
    # plt.plot(zero_array)
    #
    # plt.show()


    cubics = CubicSpline(q, I, bc_type='natural')
    # cubics = CubicSpline(q, zero_array, bc_type='natural')

    I = cubics(q2)
    # plt.plot(np.exp(I))
    # plt.show()
    # I_mat = gen_product_matrix(I)


    #standartization

    I = I[:IMAGE_DIM] #for embeddings

    # I = standartization(I)
    # I /= np.max(I)
    return 4*I

class Processing:
    def __init__(self, custom_dataraw_folder=None, number_to_process=None, start_to_process=None):
        self.custom_folder = custom_dataraw_folder
        filepath = os.path.dirname(__file__)
        self.blanks = np.load(os.path.join(filepath, 'blanks_raw.npy'))
        self.num = number_to_process
        self.start = start_to_process

        if custom_dataraw_folder is None:
            self.save_path_folder = os.path.join(filepath, 'Synthetic_Processed')
            self.load_path_folder = os.path.join(filepath, 'Synthetic_raw')

            print(self.save_path_folder)
            # print(self.blanks)
            print(self.load_path_folder)
        else:
            # self.load_path_folder = self.custom_folder
            self.load_path_folder = os.path.join(os.getcwd(), self.custom_folder)
            self.save_path_folder = os.path.join(self.load_path_folder, 'Synthetic_Processed')

            if not os.path.exists(self.save_path_folder):
                os.mkdir(self.save_path_folder)

            print(self.save_path_folder)
            # print(self.blanks)
            print(self.load_path_folder)

        self.paths_generator = self.get_filenames_without_ext(self.load_path_folder)

    def get_filenames_without_ext(self, folder_path):
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                name, extension = os.path.splitext(filename)
                if (name != '.DS_Store') and (name != 'README'):
                    yield name

    def process(self):

        if self.paths_generator is not None:
            for raw_data_name in self.paths_generator:
                print(raw_data_name)
                raw_data_name_extensioned = '{}.npy'.format(raw_data_name)
                load_path = os.path.join(self.load_path_folder, raw_data_name_extensioned)

                if 'cubic' in raw_data_name:
                    rawdata = np.load(load_path)
                    if self.num is not None:
                        rawdat = [i for i in rawdata[self.start:self.start+self.num]]
                    else: rawdat = [i for i in rawdata]
                    print('Processing {} cubic...'.format(raw_data_name))
                    processed_cubic = parallel_process(rawdat, Process_cubic)
                    processed_cubic = np.array(processed_cubic, dtype=np.float16)

                    processed_data_name_extensioned = '{}_cubic_processed.npy'.format(raw_data_name_extensioned[:4])
                    print(processed_data_name_extensioned)

                    save_path = os.path.join(self.save_path_folder, processed_data_name_extensioned)
                    print(save_path, "{} SAVING".format(raw_data_name))
                    # np.savez_compressed(save_path, processed_cubic)
                    np.save(save_path, processed_cubic)
                if 'lamellar' in raw_data_name:
                    pass
                if 'hexagonal' in raw_data_name:
                    pass

                save_path = os.path.join(self.save_path_folder, 'cubic_q.npy')
                np.save(save_path, np.linspace(0.001, 0.2, 500))
