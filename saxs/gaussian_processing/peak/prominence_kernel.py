import csv

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, medfilt

from saxs.gaussian_processing.functions import moving_average
from saxs.gaussian_processing.peak.abstract_kernel import AbstractPeakKernel
from saxs.gaussian_processing.peak.default_kernel import DefaultPeakKernel
from saxs.gaussian_processing.settings_processing import START


class ProminencePeakKernel(DefaultPeakKernel):
    str_type = 'prominence_kernel'
    short_str_type = 'prom_kern'

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

        self.peaks = None

    def preprocessing(self):
        self.default_preprocessing()
        self.detecting_relevant_noisy()

        # print(len(self.current_q_state))
        # print(len(self.current_I_state))
        # print(len(self.dI))

    def filtering(self):
        self.filtering_decomposition()

    def detecting_relevant_noisy(self):
        self.peaks, self.props = find_peaks(self.current_I_state, height=1, prominence=1)
        if len(self.props["right_bases"]) > 0:
            self.noisy_relevant_cut_point = self.props["right_bases"][0]
        else:
            self.noisy_relevant_cut_point = 100

    def filtering_decomposition(self):
        # noisy_part = np.ones(noisy_indice)
        self.detecting_relevant_noisy()

        noisy_part = moving_average(self.current_I_state[:self.noisy_relevant_cut_point], 10)

        # noiseless_part = medfilt(self.I_background_reduced[noisy_indice:], 3)
        noiseless_part = self.current_I_state[self.noisy_relevant_cut_point:]

        self.current_I_state = medfilt(np.concatenate((noisy_part, noiseless_part)), 3)

        self.total_fit = np.zeros_like(self.current_I_state)

    def search_peaks(self, height=1, prominence=0.3, distance=10):
        self.peaks, self.props = find_peaks(self.current_I_state,
                                            height=height,
                                            prominence=prominence, distance=distance)

        # print(self.props)

        # print(self.props['left_bases'])
        # print(self.props['right_bases'])
        # print(self.peaks)
        # print(self.current_q_state[self.peaks])
        # self.current_state_plot()
        # self.peaks_plots()

    def custom_sample_postprocessing(self):
        pass

        # background reduction
        # with open('without_back_res/{}'.format(self.filename), mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(np.stack((self.current_q_state, self.current_I_state), axis=1))

    def gathering(self) -> dict:
        peak_number = len(self.peaks) if self.peaks is not None else -1

        return {
            'peak_number': peak_number,
            'q': self.current_q_state[self.peaks].tolist(),
            'peaks': self.peaks.tolist(),
            # 'I': self.current_I_state[self.peaks].tolist(),

            # 'kernel': self.str_type
            # 'dI': dI.tolist(),
            # 'I_raw': I_raw.tolist(),
            # 'peaks': peaks_detected.tolist(),
            # 'params_amplitude': self.params.tolist()[::3],
            # 'params_mean': self.params.tolist()[1::3],
            # 'params_sigma': self.params.tolist()[2::3],
            # 'start_loss': self.start_loss,
            # 'final_loss': self.final_loss,
            # 'error': error
            # 'loss_ratio': self.final_loss / self.start_loss
        }


class SyntheticPeakKernel(ProminencePeakKernel):
    def __init__(self, data_dir,
                 is_preprocessing=False,
                 is_background_reduction=False,
                 is_filtering=False,
                 ):
        super().__init__(data_dir,
                         is_preprocessing,
                         is_background_reduction,
                         is_filtering,
                         )


class RobustProminencePeak(ProminencePeakKernel):
    def __init__(self, data_dir,
                 is_preprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 ):
        super().__init__(data_dir,
                         is_preprocessing,
                         is_background_reduction,
                         is_filtering,
                         )
