import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from saxs.gaussian_processing.functions import background_hyberbole
from saxs.gaussian_processing.peak.abstract_kernel import AbstractPeakKernel
from saxs.gaussian_processing.settings_processing import BACKGROUND_COEF, START


class DefaultPeakKernel(AbstractPeakKernel):
    str_type = 'default_kernel'
    short_str_type = 'def_kern'

    def __init__(self, data_dir, file_analysis_dir,
                 is_preprocessing=True,
                 is_postprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 is_peak_processing=True,
                 ):
        super().__init__(data_dir,
                         file_analysis_dir,
                         is_preprocessing,
                         is_postprocessing,
                         is_background_reduction,
                         is_filtering,
                         is_peak_processing
                         )

        self.q_cut = None
        self.I_cut = None

        self.noisy_relevant_cut_point = 0
        self.noisy_irrelevant_cut_point = 0

    def background_reduction(self):
        self.default_background_reduction()

    def preprocessing(self):
        self.default_preprocessing()

    def default_background_reduction(self, background_function=background_hyberbole):
        # self.peaks_plots = np.zeros((20, len(self.q)))

        popt, pcov = curve_fit(
            f=background_function,
            xdata=self.current_q_state,
            ydata=self.current_I_state,  # TODO after of before preprocesss?
            p0=(3, 2),
            sigma=self.dI
        )

        self.popt_background = popt
        self.pcov_background = pcov

        self.background = background_hyberbole(self.current_q_state, self.popt_background[0], self.popt_background[1])

        self.current_I_state = self.current_I_state - BACKGROUND_COEF * self.background

        self.I_background_filtered = self.current_I_state

    def default_preprocessing(self):
        self.cutting_irrelevant_noisy()

    def cutting_irrelevant_noisy(self):
        self.noisy_irrelevant_cut_point = np.argmax(self.q_raw > START)
        self.current_q_state, self.current_I_state = self.q_raw[self.noisy_irrelevant_cut_point:], \
            self.I_raw[self.noisy_irrelevant_cut_point:],

        if self.dI is not None:
            self.dI = self.dI[self.noisy_irrelevant_cut_point:]
            # print(len(self.dI), 'len dI')

        self.q_cut, self.I_cut = self.current_q_state, self.current_I_state
        self.max_I = np.max(self.current_I_state)

    # def default_filtering(self):
    #     self.difference = savgol_filter(I - background_coef * self.saxs_model, 15, 4, deriv=0)
    #     self.start_difference = savgol_filter(I - background_coef * self.saxs_model, 15, 4, deriv=0)
