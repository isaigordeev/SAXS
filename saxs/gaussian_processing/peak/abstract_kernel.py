import os

import numpy as np
from scipy.optimize import curve_fit

from saxs.gaussian_processing.functions import background_hyberbole
from saxs.gaussian_processing.processing_outils import read_data
import matplotlib.pyplot as plt

from saxs.gaussian_processing.settings_processing import BACKGROUND_COEF


class AbstractPeakKernel:
    __slots__ = ('I_raw',
                 'q_raw',
                 'I_cut',
                 'q_cut',
                 'dI',
                 'data_dir',
                 'current_I_state',
                 'current_q_state',
                 'is_background_reduction',
                 'is_preprocessing',
                 'is_postprocessing',
                 'is_filtering',
                 'popt_background',
                 'pcov_background',
                 'background',
                 'zero_level',
                 'total_fit',
                 'I_background_filtered',
                 'max_I',
                 'delta_q',
                 'peaks',
                 'str_type',
                 'short_str_type',
                 'file_analysis_dir',
                 'file_analysis_dir_peaks',
                 'noisy_irrelevant_cut_point',
                 'is_peak_processing',
                 'data_dir',
                 'data_path',
                 'filename',
                 'props'
                 )

    @classmethod
    def class_short_info(cls):
        return cls.short_str_type

    @classmethod
    def class_info(cls):
        return cls.str_type

    def __init__(self, data_dir,
                 file_analysis_dir,
                 is_preprocessing=True,
                 is_postprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 is_peak_processing=True,
                 ):

        self.is_peak_processing = is_peak_processing
        self.data_dir = data_dir
        self.data_path, self.filename = os.path.split(data_dir)
        self.file_analysis_dir = file_analysis_dir
        self.file_analysis_dir_peaks = os.path.join(self.file_analysis_dir, 'peaks/')
        self.q_raw, self.I_raw, self.dI = read_data(self.data_dir)

        self.noisy_irrelevant_cut_point = 0

        # print(self.q_raw)

        self.max_I = np.max(self.I_raw)
        self.delta_q = (self.q_raw[np.size(self.q_raw) - 1] - self.q_raw[0]) / np.size(self.q_raw)

        self.I_background_filtered = None
        self.zero_level = np.zeros(len(self.q_raw))
        self.total_fit = self.zero_level

        self.data_dir = data_dir

        self.current_I_state = self.I_raw
        self.current_q_state = self.q_raw

        self.is_preprocessing = is_preprocessing
        self.is_postprocessing = is_postprocessing

        self.is_background_reduction = is_background_reduction
        self.is_filtering = is_filtering

        self.background = None
        self.popt_background = None
        self.pcov_background = None

        self.str_type = 'abstract_kernel'
        self.short_str_type = 'abs_kern'

    def __call__(self, *args, **kwargs):
        self.custom_sample_preprocessing()
        self.sample_processing()
        self.custom_sample_postprocessing()
        return self.gathering()

    def __str__(self):
        print(self.short_str_type)
        return ''.join(self.short_str_type)

    def custom_sample_postprocessing(self):
        pass

    def custom_sample_preprocessing(self):
        pass

    def current_state_plot(self):
        plt.clf()
        plt.plot(self.current_q_state, self.current_I_state, label='current_state')

    def initial_state_plot(self):
        plt.clf()
        plt.plot(self.q_raw, self.I_raw, label='raw_plot')
        plt.plot(self.current_q_state, self.current_I_state, label='starting_state')
        plt.legend()
        plt.savefig("{}/starting_state.pdf".format(self.file_analysis_dir))

    def raw_plot(self):
        plt.clf()
        plt.plot(self.q_raw, self.I_raw, label='raw_plot')
        plt.plot(self.q_raw, self.zero_level, label='zero_level')

        plt.legend()
        plt.savefig("{}/raw_state.pdf".format(self.file_analysis_dir))

    def peaks_plots(self):
        plt.clf()
        plt.plot(self.current_q_state, self.current_I_state, label='current_state')
        plt.plot(self.current_q_state[self.peaks], self.current_I_state[self.peaks], 'rx', label='peaks')
        plt.plot(self.q_raw, self.zero_level, label='zero_level')
        plt.legend()
        plt.savefig("{}/peaks_plot.pdf".format(self.file_analysis_dir))

    def final_plot(self):
        plt.clf()
        plt.plot(self.q_raw, self.I_raw, label='raw_plot')
        plt.plot(self.q_raw[self.noisy_irrelevant_cut_point + self.peaks],
                 self.I_raw[self.noisy_irrelevant_cut_point + self.peaks], 'rx', label='peaks_on_raw')
        plt.plot(self.q_raw, self.zero_level, label='zero_level')
        plt.legend()
        plt.savefig("{}/final_plot.pdf".format(self.file_analysis_dir))

    def extended_peaks_plots(self):
        plt.clf()
        plt.plot(self.current_q_state[self.peaks], self.current_I_state[self.peaks], 'rx', label='peaks')
        plt.legend()

    def background_plot(self):
        plt.clf()
        if self.q_cut is not None and self.I_cut is not None:
            plt.plot(self.q_cut, self.I_cut, label='starting_state')
        else:
            plt.plot(self.q_raw, self.I_raw, label='starting_state')
        plt.plot(self.current_q_state, self.I_background_filtered, label='background_reduced')
        plt.plot(self.current_q_state, self.background, label='background')
        plt.plot(self.current_q_state, self.background * BACKGROUND_COEF, label='background_moderated')

        plt.plot(self.q_raw, self.zero_level, label='zero_level')
        plt.legend()
        plt.savefig("{}/background_state.pdf".format(self.file_analysis_dir))

    def filtering_plot(self):
        plt.clf()
        plt.plot(self.current_q_state, self.I_background_filtered, label='background_reduced')
        plt.plot(self.current_q_state, self.current_I_state, label='background_reduced_filtered')
        plt.plot(self.q_raw, self.zero_level, label='zero_level')

        plt.legend()

        plt.savefig("{}/filtered_state.pdf".format(self.file_analysis_dir))

    def preprocessing(self):
        # self.I_filt = self.I_filt[i:]
        pass

    def postprocessing(self):
        pass

    def filtering(self):
        pass

    def background_reduction(self):
        pass

    def search_peaks(self, *args):
        pass

    def gathering(self) -> dict:
        pass

    def sample_processing(self):

        self.raw_plot()

        if self.is_preprocessing:
            self.preprocessing()

        self.initial_state_plot()

        if self.is_background_reduction:
            self.background_reduction()
            self.background_plot()

        if self.is_filtering:
            self.filtering()
            self.filtering_plot()

        if self.is_peak_processing:
            self.search_peaks()
            self.peaks_plots()
            self.final_plot()

        if self.is_postprocessing:
            self.postprocessing()
