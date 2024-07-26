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

    def __init__(self, data_path,
                 kernel: AbstractPeakKernel = None,
                 is_preprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 write_data=True,
                 is_peak_processing=True,
                 ):
        super().__init__(data_path)

        self.write_data = write_data
        self.peak_classificator = None
        self.kernel = kernel
        self.file_analysis_dir_peaks = None
        self.file_analysis_dir = None

        self.is_preprocessing = is_preprocessing
        self.is_background_reduction = is_background_reduction
        self.is_filtering = is_filtering
        self.is_peak_processing = is_peak_processing
        self.samples = None

        if os.path.isdir(self.data_directory):
            self.find_files_in_directory()
        elif os.path.isfile(self.data_directory):
            filename = os.path.basename(self.data_directory)
            name, extension = os.path.splitext(filename)
            self.samples = [(name, extension)]

    def find_files_in_directory(self):
        self.samples = get_filenames_without_ext(self.data_directory)

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

    def peak_classification_run(self):
        for sample_name, sample_ext in self.samples:
            sample = '{}{}'.format(sample_name, sample_ext)  # TODO what is it?
            print(sample)
            self.set_output_peak_directories(sample)

            try:
                if os.path.isdir(self.data_directory):
                    self.peak_classificator = self.kernel(
                        os.path.join(self.data_directory, sample),
                        self.file_analysis_dir,
                        self.is_peak_processing,
                        self.is_background_reduction,
                        self.is_filtering,
                        self.is_peak_processing)
                else:
                    self.peak_classificator = self.kernel(
                        os.path.join(self.data_directory),
                        self.file_analysis_dir,
                        self.is_peak_processing,
                        self.is_background_reduction,
                        self.is_filtering,
                        self.is_peak_processing)

                self.data[sample] = self.peak_classificator()
            except Exception as e:
                print(f"Error processing sample {sample}: {e}")
                self.data[sample] = {'error': e}

        if self.write_data:
            self.write_peaks_data()
