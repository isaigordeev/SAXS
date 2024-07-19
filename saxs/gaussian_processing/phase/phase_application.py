import json
import os.path
from datetime import date, datetime

from saxs import DEFAULT_PHASES_PATH
from saxs.gaussian_processing.processing_classificator import ApplicationClassificator
from saxs.gaussian_processing.processing_outils import get_filenames_without_ext
from saxs.gaussian_processing.settings_processing import *

# from fastdtw import fastdtw


today = date.today()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")


def ratio(data: np.array) -> np.array:
    for i in range(len(data) - 1):
        data[i] = data[i + 1] / data[i]
    return data


def ratio_data(i, data: np.array) -> np.array:
    return (data / data[i])[1:]


class PhaseApplication(ApplicationClassificator):
    __slots__ = ('phases',
                 'phases_coefficients',
                 'phases_directory',
                 'data',
                 'phases_number',
                 'phases_dict')

    def __init__(self, data_directory, kernel, phases_directory=DEFAULT_PHASES_PATH):
        super().__init__(data_directory)

        self.samples = None
        self.phases_directory = phases_directory
        self.kernel = kernel

        # filepath, extension = os.path.split(data_directory)
        # assert extension == '.json'

        self.phases_coefficients = np.array([])
        self.phases = {}
        self.data = {}
        self.phases_dict = {}
        self.phases_number = 0

        self.set_phases()
        self.load_peak_data()

    def set_phases(self):
        with open(self.phases_directory, 'r') as file:  # NOTE make it better with string formatting
            self.phases = json.load(file)

        self.phases_coefficients = [[0] * len(x) for x in self.phases.values()]

        for i, phase in enumerate(self.phases.values()):
            phase = np.sqrt(phase)
            self.phases_coefficients[i] = ratio_data(0, phase)

        self.phases_number = len(self.phases.values())

        for i, phase in enumerate(self.phases.keys()):
            self.phases_dict[i] = phase


    def load_peak_data(self):

        with open(self.data_directory, 'r') as f:
            self.data = json.load(f)

    def write_phase_data(self):  # TODO MAKE STATIC

        # with open('{}.json'.format(self._current_results_dir_session), 'r') as f:
        #     directory_data = json.load(f)
        with open(self.data_directory, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(",", ": "))

    def phase_classification_run(self):

        self.samples = self.data.keys()
        for sample in self.samples:
            # sample = '{}{}'.format(sample_name, sample_ext)
            # print(id(self.data), 'class')
            try:
                phase_classificator = self.kernel(
                    sample,
                    self.data,
                    self.phases_dict,
                    self.phases_coefficients
                )

                self.data[sample]['predicted_phase'] = phase_classificator()
            except:
                self.data[sample]['predicted_phase'] = "error"
        self.write_phase_data()
