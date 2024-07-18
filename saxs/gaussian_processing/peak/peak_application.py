import json
import os
from abc import ABC

import numpy as np
import pandas as pd

from saxs.gaussian_processing.peak.abstract_kernel import AbstractPeakKernel
from saxs.gaussian_processing.processing_classificator import ApplicationClassificator
from saxs.gaussian_processing.processing_outils import get_filenames, get_filenames_without_ext
from saxs.gaussian_processing.settings_processing import EXTENSION, ANALYSE_DIR_SESSIONS, ANALYSE_DIR_SESSIONS_RESULTS


class PeakApplication(ApplicationClassificator):

    def __init__(self, data_directory,
                 kernel: AbstractPeakKernel = None,
                 is_preprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 is_peak_processing=True,
                 ):
        super().__init__(data_directory)

        self.peak_classificator = None
        self.kernel = kernel
        self.file_analysis_dir_peaks = None
        self.file_analysis_dir = None
        self.samples = get_filenames_without_ext(self.data_directory)

        self.is_preprocessing = is_preprocessing
        self.is_background_reduction = is_background_reduction
        self.is_filtering = is_filtering
        self.is_peak_processing = is_peak_processing

    def set_output_peak_directories(self, filename):  # TODO MAKE STATIC
        self.file_analysis_dir = os.path.join(self._result_plots_dir, filename)
        self.file_analysis_dir_peaks = os.path.join(self.file_analysis_dir, 'peaks')

        if not os.path.exists(self.file_analysis_dir):
            os.mkdir(self.file_analysis_dir)
        if not os.path.exists(self.file_analysis_dir_peaks):
            os.mkdir(self.file_analysis_dir_peaks)

    def write_peaks_data(self):  # TODO MAKE STATIC

        # with open('{}.json'.format(self._current_results_dir_session), 'r') as f:
        #     directory_data = json.load(f)
        with open(self._default_peak_data_path,
                  'w') as f:
            json.dump(self.data, f, indent=4, separators=(",", ": "))

    def peak_classification(self):
        for sample_name, sample_ext in self.samples:
            sample = '{}{}'.format(sample_name, sample_ext)
            self.set_output_peak_directories(sample)

            try:
                self.peak_classificator = self.kernel(
                    os.path.join(self.data_directory, sample),
                    self.file_analysis_dir,
                    self.is_peak_processing,
                    self.is_background_reduction,
                    self.is_filtering,
                    self.is_peak_processing)

                print(sample)
                self.data[sample] = self.peak_classificator()
            except:
                self.data[sample] = {'error': None}

        self.write_peaks_data()
